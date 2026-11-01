"""Optional CPU adapter: exported gsplat model weights -> portable NPZ.

Run in an environment with torch and gsoptviz installed. No trainer import is needed.
Input is the same manifest as the viewer, except frame gaussians may reference
exported .pt model dictionaries with a `splats` key (or directly their contents).
Camera files remain explicit; no hidden pose interpretation or refitting.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gsoptviz import Recorder, Sequence
from gsoptviz.sequence import poses, load_gaussians


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    manifest = Path(a.manifest).resolve()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    base = manifest.parent
    convention, pose_type = data.get("camera_convention", "opencv"), data.get("pose_type", "c2w")
    gt = poses(np.load(base / data["gt_c2w"], allow_pickle=False), convention, pose_type)
    recorder = Recorder(a.output, gt, data["K"], data["width"], data["height"], data["camera_ids"],
                        data.get("title", "Imported optimization"), data.get("model_to_gt"),
                        data.get("metadata", {}))
    for frame in data["frames"]:
        path = base / frame["gaussians"]
        if path.suffix == ".pt":
            state = torch.load(path, map_location="cpu", weights_only=True)
            s = state.get("splats", state)
            raw = {name: s[key].detach().cpu().numpy() for name, key in
                   [("means", "means"), ("log_scales", "scales"), ("quats_wxyz", "quats"),
                    ("opacity_logits", "opacities")]}
            raw["opacity_logits"] = raw["opacity_logits"].reshape(-1)
            raw["sh_dc"] = s["sh0"].detach().cpu().numpy().reshape(-1, 3)
            if "shN" in s:
                raw["sh_rest"] = s["shN"].detach().cpu().numpy()
        else:
            raw = load_gaussians(path)
        c = poses(np.load(base / frame["cameras"], allow_pickle=False), convention, pose_type)
        recorder.record(frame["step"], raw, c, frame.get("label", ""))
    print(json.dumps(Sequence(Path(a.output) / "manifest.json").summary(), indent=2))


if __name__ == "__main__":
    main()
