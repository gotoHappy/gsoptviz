"""Small NumPy recorder, usable from any optimization loop."""
import json
from pathlib import Path
import numpy as np

from .sequence import poses, validate_gaussians, similarity


class Recorder:
    def __init__(self, output, gt_c2w, K, width, height, camera_ids=None, title="Optimization", model_to_gt=None, metadata=None):
        self.root = Path(output)
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "frames").mkdir()
        gt = poses(gt_c2w)
        self.count = len(gt)
        ids = list(camera_ids) if camera_ids is not None else [str(i) for i in range(len(gt))]
        if len(ids) != len(gt) or len(set(ids)) != len(ids):
            raise ValueError("camera_ids must be unique and match GT cameras")
        similarity(model_to_gt)
        np.save(self.root / "gt_c2w.npy", gt)
        self.manifest = {"schema_version": 1, "title": title, "camera_convention": "opencv", "pose_type": "c2w",
                         "gt_c2w": "gt_c2w.npy", "K": np.asarray(K).tolist(), "width": width, "height": height,
                         "camera_ids": ids, "model_to_gt": model_to_gt, "metadata": metadata or {}, "frames": []}

    def record(self, step, gaussians, c2w, label=""):
        """gaussians: means, log_scales, quats_wxyz, opacity_logits, sh_dc; optional sh_rest."""
        if type(step) is not int or step < 0 or (self.manifest["frames"] and step <= self.manifest["frames"][-1]["step"]):
            raise ValueError("Record strictly increasing nonnegative integer steps")
        c = poses(c2w)
        if len(c) != self.count:
            raise ValueError("Camera count/order must stay consistent with camera_ids")
        raw = validate_gaussians(gaussians)
        stem = f"frames/{step:08d}"
        weights, cameras = stem + ".npz", stem + "_c2w.npy"
        with (self.root / (weights + ".tmp")).open("wb") as handle:
            np.savez_compressed(handle, **raw)
        (self.root / (weights + ".tmp")).replace(self.root / weights)
        with (self.root / (cameras + ".tmp")).open("wb") as handle:
            np.save(handle, c)
        (self.root / (cameras + ".tmp")).replace(self.root / cameras)
        self.manifest["frames"].append({"step": step, "gaussians": weights, "cameras": cameras, "label": label})
        temporary = self.root / "manifest.json.tmp"
        temporary.write_text(json.dumps(self.manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.root / "manifest.json")
