"""Viser scene shared by interactive playback and standalone animation export."""
import html
import json
from pathlib import Path
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation
import viser

from .sequence import Sequence

GT = (30, 205, 133)
CURRENT = (255, 150, 55)
TRAIL = (83, 155, 255)
ERROR = (220, 90, 132)


def frustum_segments(cameras, K, width, height, size):
    """Use actual asymmetric intrinsics, OpenCV +Z-forward, +Y-down."""
    uv = np.array([[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]], dtype=float)
    corners = np.einsum("nij,kj->nki", np.linalg.inv(K), uv) * size
    local = np.concatenate([np.zeros((len(cameras), 1, 3)), corners], axis=1)
    # Small upward triangle makes the camera roll/direction observable.
    up = np.stack([corners[:, 0] * .65 + corners[:, 1] * .35,
                   (corners[:, 0] + corners[:, 1]) / 2 + np.array([0, -size * .17, 0]),
                   corners[:, 0] * .35 + corners[:, 1] * .65], axis=1)
    local = np.concatenate([local, up], axis=1)
    world = np.einsum("nij,nkj->nki", cameras[:, :3, :3], local) + cameras[:, None, :3, 3]
    edges = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1], [5, 6], [6, 7]])
    return world[:, edges].reshape(-1, 2, 3).astype(np.float32)


class Scene:
    def __init__(self, server, sequence, max_gaussians=0):
        self.server, self.seq, self.limit = server, sequence, max_gaussians
        self.index = 0
        # Default to a camera that actually moves, instead of a fixed gauge anchor.
        self.selected = int(np.argmax(np.linalg.norm(sequence.cameras[-1, :, :3, 3] - sequence.cameras[0, :, :3, 3], axis=-1)))
        self.size = sequence.radius * .065
        self.splat_visible, self.trails_visible = True, True
        self.lock = threading.RLock()
        s = server.scene
        s.set_up_direction("+z")
        s.world_axes.visible = False
        server.initial_camera.position = tuple(sequence.center + sequence.radius * np.array([1.65, -2.0, 1.1]))
        server.initial_camera.look_at = tuple(sequence.center * np.array([1, 1, 0]))
        server.initial_camera.up = (0., 0., 1.)
        server.initial_camera.fov = .95
        self.overview_position = np.asarray(server.initial_camera.position)
        self.overview_target = np.asarray(server.initial_camera.look_at)
        self.lines("/GT_cameras_fixed", frustum_segments(sequence.gt, sequence.K, sequence.width, sequence.height, self.size), GT, 1.4)
        s.add_point_cloud("/GT_centers_fixed", sequence.gt[:, :3, 3].astype(np.float32), colors=GT, point_size=.006 * sequence.radius)
        self.info = s.add_label("/iteration", "", position=tuple(sequence.center + [0, 0, sequence.radius * .9]), font_screen_scale=1.1)
        self.gt_label = s.add_label("/GT_reference", "GT camera 0 · fixed", position=sequence.gt[0, :3, 3], font_screen_scale=.85)
        self.draw(0)

    def lines(self, name, points, color, thickness=1.):
        # Coincident GT/current centers and fixed anchors need no visible edge.
        points = points[np.linalg.norm(points[:, 1] - points[:, 0], axis=-1) > 1e-9]
        visible = bool(len(points))
        if not visible:
            points = np.array([[[0., 0., 0.], [1., 0., 0.]]], dtype=np.float32)
        return self.server.scene.add_line_segments(name, points=points, colors=color, thickness=thickness, thickness_units="screen", visible=visible)

    def draw(self, index):
        with self.lock, self.server.atomic():
            seq = self.seq
            if not 0 <= index < len(seq.frames):
                raise IndexError(index)
            data = seq.splats(index, self.limit)
            self.index = index
            if hasattr(self, "preloaded"):
                self.splat.visible = False
                self.splat = self.preloaded[index]
                self.splat.visible = self.splat_visible
            else:
                self.splat = self.server.scene.add_gaussian_splats("/gaussians/current", **{k: v for k, v in data.items() if k != "total"}, visible=self.splat_visible)
            self.lines("/current_cameras", frustum_segments(seq.cameras[index], seq.K, seq.width, seq.height, self.size), CURRENT, 1.6)
            self.lines("/translation_to_GT", np.stack([seq.gt[:, :3, 3], seq.cameras[index, :, :3, 3]], axis=1).astype(np.float32), ERROR, 1.)
            # Clear stale trails when scrubbing backwards, including back to frame 0.
            trajectory = seq.cameras[:index + 1, :, :3, 3]
            segments = np.stack([trajectory[:-1], trajectory[1:]], axis=2).reshape(-1, 2, 3).astype(np.float32)
            if not len(segments):
                segments = np.zeros((1, 2, 3), dtype=np.float32)
            self.trail = self.lines("/camera_trails", segments, TRAIL, 1.5)
            self.has_trail = bool(np.any(np.linalg.norm(segments[:, 1] - segments[:, 0], axis=-1) > 1e-9))
            self.trail.visible = self.trails_visible and self.has_trail
            position, rotation = seq.errors[index]
            count = len(data["centers"])
            self.info.text = (f"Step {seq.frames[index]['step']:,} · Gaussians {count:,}/{data['total']:,}\n"
                              f"Mean position error {position.mean():.4f} · rotation {rotation.mean():.2f}°"
                              + ("\n" + seq.frames[index]['label'] if seq.frames[index].get('label') else ""))
            self.highlight(self.selected)
        self.server.flush()

    def preload(self):
        """Keep offline GPU topology constant; playback only changes visibility.

        Viser 1.1.1's serialized playback can retain stale splat sort indices
        when buffer sizes change. Preloading avoids topology changes and makes
        seeking deterministic, at the cost of holding all snapshots in memory.
        Live playback continues to keep just the current snapshot on the GPU.
        """
        self.splat.remove()
        self.preloaded = []
        for index in range(len(self.seq.frames)):
            data = self.seq.splats(index, self.limit)
            handle = self.server.scene.add_gaussian_splats(
                f"/gaussians/frame_{index}",
                **{k: v for k, v in data.items() if k != "total"},
                visible=index == self.index and self.splat_visible,
            )
            self.preloaded.append(handle)
        self.splat = self.preloaded[self.index]
        self.server.flush()

    def highlight(self, selected):
        with self.lock, self.server.atomic():
            self.selected = selected
            seq = self.seq
            for name, camera, color in [("GT_selected", seq.gt[selected:selected+1], GT),
                                        ("current_selected", seq.cameras[self.index, selected:selected+1], CURRENT)]:
                self.lines("/" + name, frustum_segments(camera, seq.K[selected:selected+1], seq.width, seq.height, self.size * 1.8), color, 3.)
            self.gt_label.text = f"GT [{seq.ids[selected]}] · fixed"
            self.gt_label.position = seq.gt[selected, :3, 3]

    def legend(self):
        return ("🟢 **Reference cameras: fixed**\n\n🟠 Current cameras · 🔵 Camera trails · 🩷 Position error\n\n"
                "Saved snapshots; no interpolation across topology changes. SH0 color preview with ellipsoid shape and opacity.\n\n"
                "Position errors use reference-world units with one fixed alignment.")


class Player:
    def __init__(self, sequence, host="127.0.0.1", port=8080, max_gaussians=0):
        self.server = viser.ViserServer(host=host, port=port, label="GS optimization")
        self.scene = Scene(self.server, sequence, max_gaussians)
        self.seq = sequence
        gui = self.server.gui
        gui.configure_theme(dark_mode=True)
        gui.add_markdown(f"## {sequence.data.get('title', 'Optimization')}\n" + self.scene.legend())
        with gui.add_folder("Playback"):
            self.play = gui.add_checkbox("Play", initial_value=False)
            self.slider = gui.add_slider("Snapshot", min=0, max=max(1, len(sequence.frames)-1), step=1, initial_value=0, disabled=len(sequence.frames)==1)
            self.fps = gui.add_slider("Snapshots per second", min=1, max=20, step=1, initial_value=3)
            self.loop = gui.add_checkbox("Loop", initial_value=True)
            prev = gui.add_button("Previous")
            next_ = gui.add_button("Next")
            self.status = gui.add_markdown("")
        with gui.add_folder("View"):
            selected = gui.add_dropdown("Camera ID", options=tuple(str(x) for x in sequence.ids), initial_value=str(sequence.ids[self.scene.selected]))
            splats = gui.add_checkbox("Show Gaussians", initial_value=True)
            trails = gui.add_checkbox("Show camera trails", initial_value=True)
            overview = gui.add_button("Overview")
            pair_view = gui.add_button("Compare selected camera poses")
            gt_view = gui.add_button("View from reference camera")
            current_view = gui.add_button("View from current camera")
        self.slider.on_update(lambda _: self.refresh())
        prev.on_click(lambda _: self.move(-1))
        next_.on_click(lambda _: self.move(1))
        selected.on_update(lambda _: self.select(tuple(str(x) for x in sequence.ids).index(selected.value)))
        splats.on_update(lambda _: self.show_splats(splats.value))
        trails.on_update(lambda _: self.show_trails(trails.value))
        overview.on_click(lambda event: self.view(event.client, "overview"))
        pair_view.on_click(lambda event: self.view(event.client, "pair"))
        gt_view.on_click(lambda event: self.view(event.client, "gt"))
        current_view.on_click(lambda event: self.view(event.client, "current"))
        self.refresh()

    def refresh(self):
        index = min(int(self.slider.value), len(self.seq.frames) - 1)
        self.scene.draw(index)
        p, r = self.seq.errors[index]
        j = self.scene.selected
        self.status.content = (f"**Step {self.seq.frames[index]['step']:,}** · {index+1}/{len(self.seq.frames)} snapshots\n\n"
                               f"Mean position error `{p.mean():.5f}` · Mean rotation error `{r.mean():.3f}°`\n\n"
                               f"Selected camera `{self.seq.ids[j]}` : position `{p[j]:.5f}` · rotation `{r[j]:.3f}°`\n\n"
                               + self.seq.frames[index].get('label', ''))

    def move(self, delta):
        self.play.value = False
        self.slider.value = max(0, min(len(self.seq.frames)-1, int(self.slider.value) + delta))

    def select(self, selected):
        self.scene.highlight(selected)
        self.refresh()

    def show_splats(self, visible):
        self.scene.splat_visible = visible
        self.scene.splat.visible = visible

    def show_trails(self, visible):
        self.scene.trails_visible = visible
        self.scene.trail.visible = visible and self.scene.has_trail

    def view(self, client, which):
        if client is None:
            return
        if which == "overview":
            client.camera.position = self.scene.overview_position
            client.camera.look_at = self.scene.overview_target
            client.camera.up_direction = np.array([0., 0., 1.])
            return
        j = self.scene.selected
        if which == "pair":
            target = self.seq.gt[j, :3, 3]
            distance = max(self.scene.size,
                           float(np.linalg.norm(self.seq.cameras[self.scene.index, j, :3, 3] - target))) * 5
            client.camera.position = target + np.array([1., -1., .65]) * distance
            client.camera.look_at = target + self.seq.gt[j, :3, 2] * self.scene.size
            client.camera.up_direction = np.array([0., 0., 1.])
            return
        c = self.seq.gt[j] if which == "gt" else self.seq.cameras[self.scene.index, j]
        client.camera.position = c[:3, 3]
        client.camera.wxyz = Rotation.from_matrix(c[:3, :3]).as_quat()[[3, 0, 1, 2]]
        client.camera.up_direction = -c[:3, 1]
        client.camera.fov = 2 * np.arctan(self.seq.height / (2 * self.seq.K[j, 1, 1]))

    def run(self):
        try:
            while True:
                time.sleep(1 / self.fps.value)
                if self.play.value and len(self.seq.frames) > 1:
                    index = int(self.slider.value) + 1
                    if index == len(self.seq.frames):
                        if not self.loop.value:
                            self.play.value = False
                            continue
                        index = 0
                    self.slider.value = index
        finally:
            self.server.stop()


def export_html(sequence, output, fps=3., max_gaussians=0):
    if fps <= 0:
        raise ValueError("fps must be positive")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    server = viser.ViserServer(host="127.0.0.1", port=0)
    try:
        scene = Scene(server, sequence, max_gaussians)
        scene.preload()
        serializer = server.get_scene_serializer()
        for index in range(len(sequence.frames)):
            if index:
                scene.draw(index)
            serializer.insert_sleep(1. / fps)
            print(f"EXPORT {index+1}/{len(sequence.frames)} step={sequence.frames[index]['step']}", flush=True)
        document = serializer.as_html(dark_mode=True)
        # Fixed overlay survives orbiting and every animation frame; no network assets.
        legend = '''<aside id="gsoptviz-legend" style="position:fixed;left:16px;top:16px;z-index:1000;background:#111b2ded;color:#e9eef5;border:1px solid #324157;border-radius:12px;padding:14px 18px;font:14px/1.7 system-ui;max-width:410px;pointer-events:none">
<b style="font-size:17px">TITLE</b><br><span style="color:#1ecd85">● Reference cameras: fixed</span> · <span style="color:#ff9637">● Current cameras</span><br><span style="color:#539bff">● Camera trails</span> · <span style="color:#dc5a84">● Position error to reference</span><br>Drag to orbit · Scroll to zoom · Timeline to play/pause<br><small>Discrete snapshots; SH0 color preview. LIMIT<br>Snapshot STEP_FIRST → STEP_LAST · Fixed world alignment</small></aside>'''
        legend = legend.replace("TITLE", html.escape(sequence.data.get("title", "3DGS optimization")))
        legend = legend.replace("LIMIT", f"Showing up to {max_gaussians:,} Gaussians (subsampled preview)." if max_gaussians else "Showing all Gaussians.")
        legend = legend.replace("STEP_FIRST", str(sequence.frames[0]["step"])).replace("STEP_LAST", str(sequence.frames[-1]["step"]))
        # Start paused so a large scene finishes loading before the clock runs.
        # Viser 1.1.1 has no public autoplay option in its standalone exporter.
        pause_on_load = '''<script>
        (() => { const observer = new MutationObserver(() => {
          const icon = document.querySelector('svg[class*="player-pause"]');
          const button = icon && icon.closest('button');
          const time = document.querySelector('input');
          if (button && time) {
            observer.disconnect();
            button.click();
            // Pausing before Viser's first playback tick leaves an empty scene.
            // Explicit seeking applies time-zero messages while staying paused.
            const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
            for (const value of ['0.0001', '0']) {
              setter.call(time, value);
              time.dispatchEvent(new Event('input', {bubbles: true}));
            }
          }
        }); observer.observe(document.body, {childList:true, subtree:true}); })();
        </script>'''
        document = document.replace("</body>", legend + pause_on_load + "</body>")
        output.write_text(document, encoding="utf-8")
        report = {**sequence.summary(), "format": "standalone_html", "viser_version": viser.__version__,
                  "snapshot_storage": "preloaded_visibility_playback",
                  "start_paused": True,
                  "fps": fps, "max_gaussians": max_gaussians, "color_mode": "SH0_DC_preview",
                  "GT_visible_every_frame": True, "interpolate_weights": False, "source_manifest": sequence.path.name,
                  "output_bytes": output.stat().st_size}
        output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finally:
        server.stop()
    return report
