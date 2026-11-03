"""Tests for the project's own logic; no GPU, weights or PRIMA import needed."""
import numpy as np
import pandas as pd

from giraffe3d.detection import dedupe_by_mask
from giraffe3d.metrics import fit_metrics, scene_flags, summarize
from giraffe3d.tracking import Tracker


def _mask(x1, y1, x2, y2, size=(100, 100)):
    mask = np.zeros(size, dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def test_dedupe_removes_duplicate_and_partial_but_keeps_distinct():
    masks = np.stack([_mask(10, 10, 50, 90), _mask(11, 10, 50, 90), _mask(20, 20, 30, 40), _mask(60, 10, 90, 90)])
    keep = dedupe_by_mask(np.array([0.9, 0.8, 0.7, 0.6]), masks)
    assert sorted(keep.tolist()) == [0, 3]


def test_dedupe_keeps_small_giraffe_inside_big_giraffes_box():
    # Big giraffe's mask is an L shape, so the far giraffe is inside its box but not its mask.
    big = _mask(0, 0, 20, 100) | _mask(0, 80, 100, 100)
    far = _mask(50, 20, 70, 60)
    assert sorted(dedupe_by_mask(np.array([0.9, 0.8]), np.stack([big, far])).tolist()) == [0, 1]


def test_tracker_keeps_ids_through_motion_gap_and_crossing_neighbour():
    tracker = Tracker(max_misses=3)
    a, b = np.array([0.0, 0, 50, 100]), np.array([200.0, 0, 250, 100])
    first = tracker.update(np.stack([a, b]))
    assert list(first) == [1, 2]
    # Moves further than its own overlap in a single step, listed in the opposite order.
    assert list(tracker.update(np.stack([b + [0, 0, 0, 0], a + [40, 0, 40, 0]]))) == [2, 1]
    tracker.update(np.stack([b]))  # track 1 missed
    assert list(tracker.update(np.stack([a + [120, 0, 120, 0]]))) == [1]


def test_tracker_makes_new_id_for_far_away_box_and_expires_old():
    tracker = Tracker(max_misses=1)
    tracker.update(np.array([[0.0, 0, 50, 100]]))
    assert list(tracker.update(np.array([[800.0, 0, 850, 100]]))) == [2]
    tracker.update(np.zeros((0, 4)))
    tracker.update(np.zeros((0, 4)))
    assert list(tracker.update(np.array([[0.0, 0, 50, 100]]))) == [3]


def test_fit_metrics_perfect_and_too_short_mesh():
    det = _mask(40, 10, 60, 90)
    perfect = fit_metrics(det, det)
    assert perfect["iou"] == 1.0 and perfect["height_ratio"] == 1.0 and perfect["neck_recall"] == 1.0

    # Mesh misses the upper half: the neck recall should expose it.
    short = fit_metrics(_mask(40, 50, 60, 90), det)
    assert short["height_ratio"] == 0.5 and short["iou"] == 0.5 and short["neck_recall"] == 0.0


def test_fit_metrics_with_no_mesh_visible():
    result = fit_metrics(np.zeros((100, 100), bool), _mask(40, 10, 60, 90))
    assert result["iou"] == 0.0 and np.isnan(result["height_ratio"])


def test_scene_flags():
    boxes = np.array([[0, 10, 30, 60], [100, 10, 150, 60], [140, 10, 180, 60], [300, 10, 350, 60.0]])
    truncated, overlapping = scene_flags(boxes, 1000, 500)
    assert truncated.tolist() == [True, False, False, False]
    assert overlapping.tolist() == [False, True, True, False]


def test_summarize_excludes_truncated_from_clean_and_buckets_by_size():
    rows = [
        dict(frame=i, track_id=1, box_h=100.0 + 20 * i, cam_z=1000.0 / (100.0 + 20 * i), iou=0.8, height_ratio=1.0,
             width_ratio=1.0, area_ratio=1.0, neck_recall=0.9, neck_to_body=0.5, leg_to_body=0.4, truncated=False, overlapping=False)
        for i in range(12)
    ]
    rows.append({**rows[0], "frame": 99, "track_id": 2, "iou": 0.1, "truncated": True})
    summary = summarize(pd.DataFrame(rows), 13)
    assert summary["clean"]["n"] == 12 and summary["all"]["n"] == 13
    assert summary["clean_by_size"]["small (<150 px)"]["n"] == 3
    assert summary["clean_by_size"]["large (>=250 px)"]["n"] == 4
    track = summary["per_track"][0]
    assert track["track_id"] == 1 and abs(track["box_h_vs_inverse_depth_corr"] - 1) < 1e-6


def _toy_body(init=None):
    """Two collinear bones along x, skinned rigidly: joints at x=0 and x=1, tip at x=2."""
    import torch  # noqa: F401
    from giraffe3d.body import GiraffeBody

    v = np.array([[0.0, 0.2, 0], [0.0, -0.2, 0], [1.0, 0.2, 0], [1.0, -0.2, 0], [2.0, 0.2, 0], [2.0, -0.2, 0]])
    j_reg = np.array([[0.5, 0.5, 0, 0, 0, 0], [0, 0, 0.5, 0.5, 0, 0], [0, 0, 0, 0, 0.5, 0.5]])
    weights = np.array([[1.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]])
    faces = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4], [3, 5, 4]])
    return GiraffeBody(v, np.zeros((6, 3, 1)), j_reg, [-1, 0, 1], weights, faces, np.zeros(1), init=init,
                       groups={"a": (0,), "b": (1,), "c": (2,)})


def test_body_is_unchanged_at_unit_scale():
    import torch

    body = _toy_body()
    posed = body(torch.eye(3)[None], torch.eye(3).expand(1, 2, 3, 3))
    assert torch.allclose(posed[0], body.v_template, atol=1e-5)


def test_bone_scale_lengthens_along_bone_and_thins_across_it():
    body = _toy_body({"b": (3.0, 0.5)})
    verts, joints = body.rest()
    # Bone b runs from x=1 to x=2. Its child joint moves out to x=1+3; bone a's joint stays put.
    assert abs(float(joints[2, 0]) - 4.0) < 1e-5 and abs(float(joints[1, 0]) - 1.0) < 1e-5
    # Vertices on bone b are thinned across the bone, vertices on bone a are not.
    assert abs(float(verts[2, 1]) - 0.1) < 1e-5 and abs(float(verts[0, 1]) - 0.2) < 1e-5


def test_body_rotation_at_root_moves_vertices_rigidly():
    import torch

    body = _toy_body()
    rz90 = torch.tensor([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    posed = body(rz90[None], torch.eye(3).expand(1, 2, 3, 3))[0]
    # Root joint (0, 0, 0) stays; the far joint's vertices swing from +x to +y.
    assert torch.allclose(posed[4], torch.tensor([-0.2, 2.0, 0.0]), atol=1e-5)


def test_offsets_are_forced_symmetric():
    import torch

    body = _toy_body()
    with torch.no_grad():
        body.offsets[0] = torch.tensor([0.0, 0.1, 0.0])
    sym = body.symmetric_offsets()
    assert torch.allclose(sym[0], torch.tensor([0.0, 0.05, 0.0])) and torch.allclose(sym[1], torch.tensor([0.0, -0.05, 0.0]))


def test_mask_loss_prefers_mesh_matching_the_mask():
    import torch

    from giraffe3d.fit import CROP_RES, mask_loss, prepare_mask

    mask = np.zeros((200, 200), bool)
    mask[50:150, 80:120] = True
    center, size, distance, samples = prepare_mask(mask, np.array([80.0, 50, 120, 150]), np.random.default_rng(0))
    assert distance.shape == (CROP_RES, CROP_RES) and center.tolist() == [100.0, 100.0] and size == 160.0
    dist = torch.as_tensor(distance)[None, None]
    pts = torch.as_tensor(samples)[None]
    good = mask_loss(pts, dist, pts)
    shifted = pts + torch.tensor([0.3, 0.0])
    bad = mask_loss(shifted, dist, pts)
    assert good[0] < 1e-6 and good[1] < 1e-6 and bad[0] > good[0] and bad[1] > good[1]


def test_projection_centres_a_point_on_its_crop():
    import torch

    from giraffe3d.fit import project_to_crop

    # A point on the optical axis lands at the image centre, i.e. the middle of a crop centred there.
    uv = project_to_crop(torch.zeros(1, 1, 3), torch.tensor([[0.0, 0, 10]]), 100.0, (200, 100),
                         torch.tensor([[100.0, 50.0]]), torch.tensor([40.0]))
    assert torch.allclose(uv[0, 0], torch.tensor([0.5, 0.5]))


def test_bone_scale_is_held_within_bounds():
    from giraffe3d.body import GiraffeBody  # noqa: F401

    body = _toy_body({"b": (9.0, 0.01)})
    body2 = type(body)(body.v_template.numpy(), np.zeros((6, 3, 1)), body.j_regressor.numpy(), [-1, 0, 1],
                       body.weights.numpy(), body.faces, np.zeros(1), init={"b": (9.0, 0.01)},
                       groups={"a": (0,), "b": (1,), "c": (2,)}, bounds={"b": (0.5, 2.0, 0.5, 2.0)})
    assert abs(float(body2.scales()[1, 0]) - 2.0) < 1e-5 and abs(float(body2.scales()[1, 1]) - 0.5) < 1e-5


def test_indexed_adam_leaves_unsampled_instances_untouched():
    import torch

    from giraffe3d.fit import IndexedAdam

    param = torch.nn.Parameter(torch.ones(4, 2))
    optimizer = IndexedAdam([(param, 0.1)])
    for _ in range(3):
        optimizer.zero_grad()
        param.pow(2).sum().backward()
        optimizer.step(torch.tensor([0, 2]))
    assert (param[[1, 3]] == 1).all() and (param[[0, 2]] < 1).all()


def test_mask_loss_notices_a_few_runaway_vertices():
    import torch

    from giraffe3d.fit import mask_loss

    dist = torch.zeros(1, 1, 32, 32)
    dist[..., 16:] = 1.0  # the right half of the crop lies outside the mask
    inside = torch.full((1, 1000, 2), 0.25)
    runaway = inside.clone()
    runaway[0, :10, 0] = 0.9  # 1% of vertices stray outside
    samples = torch.full((1, 5, 2), 0.25)
    assert mask_loss(runaway, dist, samples)[0] > 0.15 and mask_loss(inside, dist, samples)[0] < 1e-6


def test_mean_rotations_recovers_a_rotation():
    from fit_giraffe import mean_rotations

    c, s = np.cos(0.2), np.sin(0.2)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    stack = np.stack([rz, rz])[:, None]  # (N=2, joints=1, 3, 3)
    result = mean_rotations(stack)
    assert np.allclose(result[0], rz, atol=1e-6) and np.isclose(np.linalg.det(result[0]), 1.0)
