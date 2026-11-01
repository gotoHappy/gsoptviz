"""Create a small synthetic sequence; no training data or checkpoint is needed."""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from gsoptviz import Recorder


def make_demo(output):
    rng = np.random.default_rng(7)
    count = 2400
    u = rng.uniform(0, 2 * np.pi, count)
    v = rng.uniform(0, 2 * np.pi, count)
    means = np.column_stack(((1 + .32 * np.cos(v)) * np.cos(u),
                             (1 + .32 * np.cos(v)) * np.sin(u),
                             .32 * np.sin(v)))
    colors = np.column_stack((.5 + .45 * np.cos(u),
                              .5 + .45 * np.sin(u),
                              .5 + .45 * np.cos(v)))
    gt = np.tile(np.eye(4), (8, 1, 1))
    for i, theta in enumerate(np.linspace(0, 2 * np.pi, len(gt), endpoint=False)):
        eye = np.array([3 * np.cos(theta), 3 * np.sin(theta), 1.5])
        forward = -eye / np.linalg.norm(eye)
        right = np.cross(forward, [0., 0., 1.])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        gt[i, :3, :3] = np.column_stack((right, down, forward))
        gt[i, :3, 3] = eye
    rec = Recorder(output, gt, [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
                   640, 480, [f"camera_{i}" for i in range(len(gt))],
                   title="Synthetic torus and camera trajectories",
                   metadata={"synthetic": True, "note": "Illustrative motion, not an optimization result."})
    noise = rng.normal(0, .25, means.shape)
    camera_noise = rng.normal(0, .25, (len(gt), 3))
    for frame in range(6):
        remaining = 1 - frame / 5
        # Deliberately change topology to exercise variable-size snapshots.
        n = [1200, 1600, 2000, 2400, 2200, 2400][frame]
        cameras = gt.copy()
        cameras[:, :3, 3] += camera_noise * remaining
        cameras[:, :3, :3] = Rotation.from_euler("z", remaining * .15).as_matrix() @ gt[:, :3, :3]
        rec.record(frame * 100, {
            "means": means[:n] + noise[:n] * remaining,
            "log_scales": np.full((n, 3), np.log(.035 + .025 * remaining)),
            "quats_wxyz": np.tile([1., 0., 0., 0.], (n, 1)),
            "opacity_logits": np.full(n, 2.),
            "sh_dc": (colors[:n] - .5) / .28209479177387814,
        }, cameras, label="SYNTHETIC DEMO: illustrative motion")
    return Path(output) / "manifest.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="demo_sequence", help="A new output directory")
    print(make_demo(parser.parse_args().output))
