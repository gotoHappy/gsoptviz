import json
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gsoptviz import Recorder, Sequence
from gsoptviz.sequence import poses, splat_geometry, transform_poses, similarity, load_gaussians


def weights(n=3):
    return {"means": np.arange(n*3, dtype=np.float32).reshape(n, 3) / 10,
            "log_scales": np.tile(np.log([.1, .2, .3]), (n, 1)),
            "quats_wxyz": np.tile([1., 0., 0., 0.], (n, 1)),
            "opacity_logits": np.zeros(n), "sh_dc": np.zeros((n, 3))}


def fixture_sequence(path):
    gt = np.tile(np.eye(4), (2, 1, 1))
    gt[1, 0, 3] = 1.
    rec = Recorder(path, gt, [[100, 0, 50], [0, 100, 50], [0, 0, 1]], 100, 100, camera_ids=["a", "b"])
    noisy = gt.copy(); noisy[:, 1, 3] += .2
    rec.record(0, weights(), noisy)
    rec.record(10, weights(5), gt)
    return rec, gt


def test_roundtrip_variable_topology_and_real_camera_errors(tmp_path):
    rec, gt = fixture_sequence(tmp_path / "sequence")
    seq = Sequence(rec.root / "manifest.json")
    assert seq.splats(0)["total"] == 3 and seq.splats(1)["total"] == 5
    assert seq.splats(1, 2)["centers"].shape == (2, 3)
    np.testing.assert_array_equal(seq.gt, gt)
    np.testing.assert_allclose(seq.errors[0][0], .2)
    np.testing.assert_allclose(seq.errors[1][0], 0)
    with pytest.raises(ValueError, match="increasing"):
        rec.record(10, weights(), gt)


def test_unicode_title_and_camera_ids_roundtrip(tmp_path):
    rec = Recorder(tmp_path / "sequence", np.eye(4)[None], np.eye(3), 100, 100,
                   camera_ids=["相机_0"], title="优化快照")
    rec.record(0, weights(), np.eye(4)[None], label="初始状态")
    seq = Sequence(rec.root / "manifest.json")
    assert seq.data["title"] == "优化快照"
    assert seq.ids == ["相机_0"]
    assert seq.frames[0]["label"] == "初始状态"


def test_fixed_similarity_transforms_covariance_and_cameras_together():
    rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    transform = similarity({"scale": 2., "rotation": rotation.tolist(), "translation": [1., 2., 3.]})
    raw = weights(1)
    geometry = splat_geometry(raw, transform)
    np.testing.assert_allclose(geometry["covariances"][0], np.diag([.16, .04, .36]), atol=1e-7)
    cam = np.eye(4)[None]
    cam[:, :3, 3] = raw["means"]
    converted = transform_poses(cam, transform)
    np.testing.assert_allclose(converted[:, :3, 3], geometry["centers"], atol=1e-7)
    np.testing.assert_allclose(converted[0, :3, :3], rotation)
    # Frame conversion does not change GT or refit a different transform per frame.
    np.testing.assert_array_equal(cam[:, :3, 3], raw["means"])


def test_pose_conventions_and_invalid_rotations():
    camera = np.eye(4)[None]; camera[0, :3, 3] = [1., 2., 3.]
    gl = camera @ np.diag([1, -1, -1, 1])
    np.testing.assert_allclose(poses(np.linalg.inv(gl), "opengl", "w2c"), camera)
    camera[0, 0, 0] = -1
    with pytest.raises(ValueError, match="proper"):
        poses(camera)


def test_missing_frame_and_mismatched_cameras_rejected(tmp_path):
    rec, _ = fixture_sequence(tmp_path / "sequence")
    manifest = rec.root / "manifest.json"
    content = json.loads(manifest.read_text())
    content["frames"][1]["step"] = 0
    manifest.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="increasing"):
        Sequence(manifest)
    manifest.write_text(json.dumps(rec.manifest))
    np.save(rec.root / rec.manifest["frames"][1]["cameras"], np.eye(4)[None])
    with pytest.raises(ValueError, match="count"):
        Sequence(manifest)


def test_standard_ply_and_npz_agree(tmp_path):
    from plyfile import PlyData, PlyElement
    raw = weights(3)
    fields = ["x", "y", "z"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)] + ["opacity"] + [f"f_dc_{i}" for i in range(3)]
    a = np.empty(3, dtype=[(field, "f4") for field in fields])
    values = np.concatenate([raw["means"], raw["log_scales"], raw["quats_wxyz"], raw["opacity_logits"][:, None], raw["sh_dc"]], axis=1)
    for i, field in enumerate(fields):
        a[field] = values[:, i]
    PlyData([PlyElement.describe(a, "vertex")]).write(tmp_path / "model.ply")
    np.savez(tmp_path / "model.npz", **raw)
    ply, npz = load_gaussians(tmp_path / "model.ply"), load_gaussians(tmp_path / "model.npz")
    for key in ply:
        np.testing.assert_allclose(ply[key], npz[key])


def test_frustum_uses_principal_point_and_opencv_direction():
    from gsoptviz.viewer import frustum_segments
    c = np.eye(4)[None]
    k = np.array([[[100, 0, 25], [0, 100, 30], [0, 0, 1]]])
    edges = frustum_segments(c, k, 100, 80, 2)
    np.testing.assert_allclose(edges[0, 0], [0, 0, 0])
    np.testing.assert_allclose(edges[0, 1], [-.5, -.6, 2.])
    np.testing.assert_allclose(edges[2, 1], [1.5, 1., 2.])


def test_player_initialization_and_backward_scrubbing_preserve_gt(tmp_path):
    from gsoptviz.viewer import Player
    rec, gt = fixture_sequence(tmp_path / "sequence")
    seq = Sequence(rec.root / "manifest.json")
    player = Player(seq, port=0)
    try:
        player.scene.draw(1)
        assert "Step 10" in player.scene.info.text
        player.scene.draw(0)
        assert "Step 0" in player.scene.info.text
        np.testing.assert_array_equal(seq.gt, gt)
        player.show_splats(False)
        player.scene.draw(1)
        assert not player.scene.splat.visible
        assert player.scene.gt_label.visible
        assert "Reference cameras: fixed" in player.scene.legend()
        player.show_splats(True)
        player.scene.preload()
        player.scene.draw(1)
        assert [h.visible for h in player.scene.preloaded] == [False, True]
        player.scene.draw(0)
        assert [h.visible for h in player.scene.preloaded] == [True, False]
        np.testing.assert_array_equal(seq.gt, gt)
    finally:
        player.server.stop()
