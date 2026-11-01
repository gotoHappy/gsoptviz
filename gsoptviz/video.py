"""Optional MP4 export through a CPU WebGL browser; same scene as the player."""
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from .viewer import Scene


def export_video(sequence, output, fps=3., max_gaussians=0, browser=None, width=1280, height=800, view="overview", camera_id=None):
    import imageio_ffmpeg
    from playwright.sync_api import sync_playwright
    import viser

    if fps <= 0 or width % 2 or height % 2:
        raise ValueError("Positive fps and even video dimensions required")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    server = viser.ViserServer(host="127.0.0.1", port=0)
    scene = Scene(server, sequence, max_gaussians)
    if camera_id is not None:
        scene.highlight([str(x) for x in sequence.ids].index(str(camera_id)))
    if view == "camera":
        j = scene.selected
        target = sequence.gt[j, :3, 3]
        distance = max(scene.size, float(np.linalg.norm(sequence.cameras[:, j, :3, 3] - target, axis=-1).max())) * 5
        server.initial_camera.position = target + np.array([1., -1., .65]) * distance
        server.initial_camera.look_at = target + sequence.gt[j, :3, 2] * scene.size
    elif view != "overview":
        server.stop()
        raise ValueError("view must be overview or camera")
    # Canvas text textures load asynchronously. Burn authoritative step text into
    # the video instead of capturing a possibly stale in-scene text texture.
    scene.info.visible = False
    ack = server.gui.add_number("Frame acknowledgement", initial_value=-1, disabled=True)
    encoder = None
    temporary = output.with_name(output.stem + ".partial.mp4")
    try:
        with sync_playwright() as p:
            options = {"headless": True, "args": ["--no-sandbox", "--disable-dev-shm-usage", "--use-gl=angle",
                                                    "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]}
            if browser:
                options["executable_path"] = browser
            chrome = p.chromium.launch(**options)
            page = chrome.new_page(viewport={"width": width, "height": height})
            page.goto(f"http://127.0.0.1:{server.get_port()}")
            deadline = time.monotonic() + 45
            while not server.get_clients():
                if time.monotonic() > deadline:
                    raise TimeoutError("Browser did not connect")
                page.wait_for_timeout(100)
            client = next(iter(server.get_clients().values()))
            # Allow initial shader compilation and asynchronous splat sorting.
            page.wait_for_timeout(1500)
            encoder = imageio_ffmpeg.write_frames(str(temporary), (width, height), fps=fps, codec="libx264",
                                                  pix_fmt_in="rgb24", pix_fmt_out="yuv420p", macro_block_size=1)
            encoder.send(None)
            for index, frame in enumerate(sequence.frames):
                scene.draw(index)
                ack.value = index
                server.flush()
                page.wait_for_function("i => {const e = document.querySelector('input:disabled'); return e && e.value === String(i);}", arg=index, timeout=30000)
                page.wait_for_timeout(800)
                rgb = client.get_render(height=height, width=width, transport_format="png", timeout=45)
                image = Image.fromarray(rgb[:, :, :3])
                draw = ImageDraw.Draw(image)
                # Legend is burned into every video frame, including the first/last.
                draw.rectangle((12, 12, 720, 112), fill=(16, 26, 44))
                draw.text((26, 24), f"3DGS optimization | step {frame['step']:,} | snapshot {index+1}/{len(sequence.frames)}", fill="white", font_size=22)
                draw.text((26, 55), "GT cameras: fixed", fill=(30, 205, 133), font_size=19)
                draw.text((230, 55), "Current cameras", fill=(255, 150, 55), font_size=19)
                draw.text((430, 55), "Camera trajectories", fill=(83, 155, 255), font_size=19)
                note = frame.get('label') or "SH0 color preview | fixed world alignment | real saved snapshots"
                draw.text((26, 84), note, fill=(200, 211, 225), font_size=16)
                pos, rot = sequence.errors[index]
                j = scene.selected
                draw.rectangle((12, height - 55, 880, height - 12), fill=(16, 26, 44))
                draw.text((26, height - 45), f"Camera {sequence.ids[j]} | position error {pos[j]:.5f} | rotation error {rot[j]:.3f} deg", fill="white", font_size=20)
                encoder.send(np.asarray(image).tobytes())
                print(f"VIDEO {index+1}/{len(sequence.frames)}", flush=True)
            encoder.close(); encoder = None
            chrome.close()
        temporary.replace(output)
        report = {**sequence.summary(), "format": "mp4", "fps": fps, "width": width, "height": height,
                  "view": view, "selected_camera_id": str(sequence.ids[scene.selected]),
                  "max_gaussians": max_gaussians, "color_mode": "SH0_DC_preview", "GT_visible_every_frame": True}
        output.with_suffix(".video.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        if encoder is not None:
            encoder.close()
        server.stop()
