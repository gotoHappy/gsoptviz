"""Versioned, framework-independent input and common-world geometry."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.spatial.transform import Rotation


def poses(value, convention="opencv", pose_type="c2w"):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 3 or a.shape[1:] != (4, 4) or not len(a) or not np.isfinite(a).all():
        raise ValueError("Cameras must be finite (C,4,4) matrices")
    if not np.allclose(a[:, 3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError("Invalid homogeneous camera row")
    r = a[:, :3, :3]
    if not np.allclose(r @ r.transpose(0, 2, 1), np.eye(3), atol=2e-4) or not np.allclose(np.linalg.det(r), 1, atol=2e-4):
        raise ValueError("Camera rotations must be proper orthonormal matrices; remove scale from poses")
    if pose_type not in ("c2w", "w2c") or convention not in ("opencv", "opengl"):
        raise ValueError("Use pose_type c2w/w2c and convention opencv/opengl")
    if pose_type == "w2c":
        a = np.linalg.inv(a)
    if convention == "opengl":
        a = a @ np.diag([1., -1., -1., 1.])
    return a.copy()


def similarity(data):
    data = data or {}
    s = float(data.get("scale", 1.))
    r = np.asarray(data.get("rotation", np.eye(3)), dtype=float)
    t = np.asarray(data.get("translation", [0, 0, 0]), dtype=float)
    if not np.isfinite(s) or s <= 0 or r.shape != (3, 3) or t.shape != (3,) or not np.isfinite(t).all():
        raise ValueError("Invalid fixed model_to_gt similarity")
    if not np.allclose(r @ r.T, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(r), 1):
        raise ValueError("Similarity rotation must be a proper rotation")
    return s, r, t


def transform_poses(a, transform):
    s, r, t = transform
    b = a.copy()
    b[:, :3, :3] = r @ a[:, :3, :3]
    b[:, :3, 3] = s * (a[:, :3, 3] @ r.T) + t
    return b


def camera_errors(current, gt):
    relative = current[:, :3, :3] @ gt[:, :3, :3].transpose(0, 2, 1)
    angle = np.rad2deg(np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    translation = np.linalg.norm(current[:, :3, 3] - gt[:, :3, 3], axis=-1)
    return translation, angle


def load_gaussians(path):
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            raw = {k: data[k] for k in ("means", "log_scales", "quats_wxyz", "opacity_logits", "sh_dc")}
    elif path.suffix.lower() == ".ply":
        from plyfile import PlyData
        v = PlyData.read(path)["vertex"]
        raw = {"means": np.stack([v[k] for k in ("x", "y", "z")], -1),
               "log_scales": np.stack([v[f"scale_{i}"] for i in range(3)], -1),
               "quats_wxyz": np.stack([v[f"rot_{i}"] for i in range(4)], -1),
               "opacity_logits": np.asarray(v["opacity"]),
               "sh_dc": np.stack([v[f"f_dc_{i}"] for i in range(3)], -1)}
    else:
        raise ValueError(f"Unsupported weights: {path.suffix}; use standard 3DGS PLY or canonical NPZ")
    return validate_gaussians(raw)


def validate_gaussians(raw):
    raw = {k: np.asarray(v, dtype=np.float32) for k, v in raw.items()}
    n = len(raw["means"])
    expected = {"means": (n, 3), "log_scales": (n, 3), "quats_wxyz": (n, 4),
                "opacity_logits": (n,), "sh_dc": (n, 3)}
    if n == 0:
        raise ValueError("Empty Gaussian frame")
    for key, shape in expected.items():
        if raw[key].shape != shape or not np.isfinite(raw[key]).all():
            raise ValueError(f"{key}: expected finite {shape}, got {raw[key].shape}")
    if np.any(np.linalg.norm(raw["quats_wxyz"], axis=-1) < 1e-8):
        raise ValueError("Zero Gaussian quaternion")
    if np.any(np.abs(raw["log_scales"]) > 40):
        raise ValueError("Extreme log-scales cannot be displayed reliably")
    return raw


def splat_geometry(raw, transform, max_gaussians=0):
    count = len(raw["means"])
    indices = np.arange(count) if max_gaussians <= 0 or count <= max_gaussians else np.linspace(0, count - 1, max_gaussians, dtype=int)
    s, world_r, t = transform
    q = raw["quats_wxyz"][indices]
    r = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    axes = world_r @ (r * np.exp(raw["log_scales"][indices])[:, None, :]) * s
    return {"centers": (s * (raw["means"][indices] @ world_r.T) + t).astype(np.float32),
            "covariances": (axes @ axes.transpose(0, 2, 1)).astype(np.float32),
            "rgbs": np.clip(.5 + .28209479177387814 * raw["sh_dc"][indices], 0, 1),
            "opacities": expit(raw["opacity_logits"][indices, None]), "total": count}


class Sequence:
    def __init__(self, manifest):
        self.path = Path(manifest).resolve()
        self.data = json.loads(self.path.read_text(encoding="utf-8"))
        if self.data.get("schema_version") != 1:
            raise ValueError("Unsupported manifest schema_version")
        self.frames = self.data["frames"]
        steps = [f["step"] for f in self.frames]
        if not steps or any(type(s) is not int or s < 0 for s in steps) or any(a >= b for a, b in zip(steps, steps[1:])):
            raise ValueError("Frame steps must be nonnegative integers in strictly increasing order")
        self.convention = self.data.get("camera_convention", "opencv")
        self.pose_type = self.data.get("pose_type", "c2w")
        self.gt = poses(np.load(self.resolve(self.data["gt_c2w"]), allow_pickle=False), self.convention, self.pose_type)
        self.ids = self.data["camera_ids"]
        if len(self.ids) != len(self.gt) or len(set(self.ids)) != len(self.ids):
            raise ValueError("camera_ids must be unique and match the pose array order")
        self.transform = similarity(self.data.get("model_to_gt"))
        self.width, self.height = int(self.data["width"]), int(self.data["height"])
        if min(self.width, self.height) <= 0:
            raise ValueError("Positive image dimensions required")
        k = np.asarray(self.data["K"], dtype=float)
        self.K = np.broadcast_to(k, (len(self.gt), 3, 3)).copy()
        if not np.isfinite(self.K).all() or np.any(self.K[:, [0, 1], [0, 1]] <= 0) or not np.allclose(self.K[:, 2], [0, 0, 1]):
            raise ValueError("Invalid pinhole intrinsics")
        self.cameras = []
        self.errors = []
        for f in self.frames:
            if not self.resolve(f["gaussians"]).is_file():
                raise FileNotFoundError(self.resolve(f["gaussians"]))
            c = poses(np.load(self.resolve(f["cameras"]), allow_pickle=False), self.convention, self.pose_type)
            if c.shape != self.gt.shape:
                raise ValueError(f"Camera count changed at step {f['step']}")
            c = transform_poses(c, self.transform)
            self.cameras.append(c)
            self.errors.append(camera_errors(c, self.gt))
        self.cameras = np.stack(self.cameras)
        self.center = np.median(self.gt[:, :3, 3], axis=0)
        self.radius = max(float(np.linalg.norm(self.gt[:, :3, 3] - self.center, axis=1).max()), 1e-2)

    def resolve(self, path):
        return self.path.parent / path

    @lru_cache(maxsize=2)
    def splats(self, index, max_gaussians=0):
        return splat_geometry(load_gaussians(self.resolve(self.frames[index]["gaussians"])), self.transform, max_gaussians)

    def summary(self):
        return {"frames": len(self.frames), "first_step": self.frames[0]["step"], "last_step": self.frames[-1]["step"],
                "cameras": len(self.ids), "fixed_alignment": True,
                "first_translation_mean": float(self.errors[0][0].mean()), "last_translation_mean": float(self.errors[-1][0].mean()),
                "first_rotation_deg_mean": float(self.errors[0][1].mean()), "last_rotation_deg_mean": float(self.errors[-1][1].mean())}
