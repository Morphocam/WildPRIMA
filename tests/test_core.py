"""Tests for the project's own logic; no GPU, weights or PRIMA import needed."""
import numpy as np
import pandas as pd
import pytest
import torch
from smplx.lbs import batch_rodrigues

from animal3d.detection import dedupe_by_mask
from animal3d.metrics import fit_metrics, scene_flags, summarize
from animal3d.sources import Camera, move_to_camera, open_source, scaled_camera
from animal3d.tracking import Tracker


def _mask(x1, y1, x2, y2, size=(100, 100)):
    mask = np.zeros(size, dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def test_dedupe_removes_duplicate_and_partial_but_keeps_distinct():
    masks = np.stack([_mask(10, 10, 50, 90), _mask(11, 10, 50, 90), _mask(20, 20, 30, 40), _mask(60, 10, 90, 90)])
    keep = dedupe_by_mask(np.array([0.9, 0.8, 0.7, 0.6]), masks)
    assert sorted(keep.tolist()) == [0, 3]


def test_dedupe_keeps_small_animal_inside_big_animals_box():
    # Big animal's mask is an L shape, so the far animal is inside its box but not its mask.
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
    assert perfect["iou"] == 1.0 and perfect["height_ratio"] == 1.0 and perfect["top_recall"] == 1.0

    # Mesh misses the upper half: the neck recall should expose it.
    short = fit_metrics(_mask(40, 50, 60, 90), det)
    assert short["height_ratio"] == 0.5 and short["iou"] == 0.5 and short["top_recall"] == 0.0


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
             width_ratio=1.0, area_ratio=1.0, top_recall=0.9, neck_to_body=0.5, leg_to_body=0.4, truncated=False, overlapping=False)
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
    from animal3d.body import AnimalBody

    v = np.array([[0.0, 0.2, 0], [0.0, -0.2, 0], [1.0, 0.2, 0], [1.0, -0.2, 0], [2.0, 0.2, 0], [2.0, -0.2, 0]])
    j_reg = np.array([[0.5, 0.5, 0, 0, 0, 0], [0, 0, 0.5, 0.5, 0, 0], [0, 0, 0, 0, 0.5, 0.5]])
    weights = np.array([[1.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]])
    faces = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4], [3, 5, 4]])
    return AnimalBody(v, np.zeros((6, 3, 1)), j_reg, [-1, 0, 1], weights, faces, np.zeros(1), init=init,
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

    from animal3d.fit import CROP_RES, mask_loss, prepare_mask

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

    from animal3d.fit import project_to_crop

    # A point on the optical axis lands at the image centre, i.e. the middle of a crop centred there.
    uv = project_to_crop(torch.zeros(1, 1, 3), torch.tensor([[0.0, 0, 10]]), torch.tensor([[100.0, 100.0, 100.0, 50.0]]),
                         torch.tensor([[100.0, 50.0]]), torch.tensor([40.0]))
    assert torch.allclose(uv[0, 0], torch.tensor([0.5, 0.5]))


def test_bone_scale_is_held_within_bounds():
    from animal3d.body import AnimalBody  # noqa: F401

    body = _toy_body({"b": (9.0, 0.01)})
    body2 = type(body)(body.v_template.numpy(), np.zeros((6, 3, 1)), body.j_regressor.numpy(), [-1, 0, 1],
                       body.weights.numpy(), body.faces, np.zeros(1), init={"b": (9.0, 0.01)},
                       groups={"a": (0,), "b": (1,), "c": (2,)}, bounds={"b": (0.5, 2.0, 0.5, 2.0)})
    assert abs(float(body2.scales()[1, 0]) - 2.0) < 1e-5 and abs(float(body2.scales()[1, 1]) - 0.5) < 1e-5


def test_indexed_adam_leaves_unsampled_instances_untouched():
    import torch

    from animal3d.fit import IndexedAdam

    param = torch.nn.Parameter(torch.ones(4, 2))
    optimizer = IndexedAdam([(param, 0.1)])
    for _ in range(3):
        optimizer.zero_grad()
        param.pow(2).sum().backward()
        optimizer.step(torch.tensor([0, 2]))
    assert (param[[1, 3]] == 1).all() and (param[[0, 2]] < 1).all()


def test_mask_loss_notices_a_few_runaway_vertices():
    import torch

    from animal3d.fit import mask_loss

    dist = torch.zeros(1, 1, 32, 32)
    dist[..., 16:] = 1.0  # the right half of the crop lies outside the mask
    inside = torch.full((1, 1000, 2), 0.25)
    runaway = inside.clone()
    runaway[0, :10, 0] = 0.9  # 1% of vertices stray outside
    samples = torch.full((1, 5, 2), 0.25)
    assert mask_loss(runaway, dist, samples)[0] > 0.15 and mask_loss(inside, dist, samples)[0] < 1e-6


def test_mean_rotations_recovers_a_rotation():
    from run import mean_rotations

    c, s = np.cos(0.2), np.sin(0.2)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    stack = np.stack([rz, rz])[:, None]  # (N=2, joints=1, 3, 3)
    result = mean_rotations(stack)
    assert np.allclose(result[0], rz, atol=1e-6) and np.isclose(np.linalg.det(result[0]), 1.0)


def _write_images(folder, names, shape=(40, 60, 3)):
    import cv2

    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        cv2.imwrite(str(folder / name), np.zeros(shape, np.uint8))


def test_open_source_recognises_each_input_type(tmp_path):
    import cv2

    _write_images(tmp_path / "photos", ["b10.jpg", "b2.jpg", "notes.txt.jpg"])
    source = open_source(tmp_path / "photos", "goat")
    assert source.kind == "images" and not source.sequence and source.species == "goat"
    assert [p.name for p in source.images] == ["b2.jpg", "b10.jpg", "notes.txt.jpg"]  # natural order

    single = open_source(tmp_path / "photos" / "b2.jpg")
    assert single.kind == "image" and single.species == "animal" and [f.name for f in single] == ["b2"]

    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (60, 40))
    for _ in range(3):
        writer.write(np.zeros((40, 60, 3), np.uint8))
    writer.release()
    source = open_source(video)
    assert source.kind == "video" and source.sequence and len(list(source)) == 3


def _snippet(tmp_path, source_line="source: July08", raw=5000):
    import cv2

    root = tmp_path / "giraffe_4"
    _write_images(root / "rgb", ["100.jpg"])
    (root / "depthm").mkdir()
    depth = np.zeros((40, 60), np.uint16)
    depth[10, 10] = raw
    cv2.imwrite(str(root / "depthm" / "100.png"), depth)
    (root / "metadata.txt").write_text(f"species: giraffe\n{source_line}\n")
    return root


def test_snippet_reads_depth_with_the_units_of_its_site(tmp_path):
    root = _snippet(tmp_path)
    source = open_source(root)
    assert source.kind == "sequence" and source.has_lidar and source.species == "giraffe"
    assert abs(next(iter(source)).depth_m[10, 10] - 5.0) < 1e-5  # Longleat: millimetres

    (root / "metadata.txt").write_text("species: giraffe\nsource: Zimbabwe\n")
    assert abs(next(iter(open_source(root))).depth_m[10, 10] - 50.0) < 1e-4  # Bubye: centimetres

    # Choosing the rgb folder itself must still find the LiDAR beside it.
    assert open_source(root / "rgb").has_lidar


def test_depth_is_ignored_without_a_known_calibration(tmp_path):
    root = _snippet(tmp_path, "source: somewhere")
    source = open_source(root)
    assert not source.has_lidar and next(iter(source)).depth_m is None


def test_folder_of_subfolders_gets_a_helpful_error(tmp_path):
    import pytest

    (tmp_path / "giraffe_4").mkdir()
    with pytest.raises(SystemExit, match="giraffe_4"):
        open_source(tmp_path)


def test_move_to_camera_keeps_the_animal_at_the_same_pixel():
    cam_t = np.array([[0.4, -0.3, 50.0]])
    focal, width, height = 7500.0, 1920, 1080
    camera = Camera(900.0, 1070.0, 845.0, 549.0)
    moved = move_to_camera(cam_t, focal, width, height, camera)[0]
    u = camera.fx * moved[0] / moved[2] + camera.cx
    v = camera.fy * moved[1] / moved[2] + camera.cy
    assert abs(u - (focal * 0.4 / 50 + width / 2)) < 1e-6 and abs(v - (focal * -0.3 / 50 + height / 2)) < 1e-6
    # The same camera changes nothing.
    same = move_to_camera(cam_t, focal, width, height, Camera(focal, focal, width / 2, height / 2))
    assert np.allclose(same, cam_t)


def test_scaled_camera_follows_the_frame_size():
    from animal3d.sources import CALIBRATIONS

    half = scaled_camera(CALIBRATIONS["longleat_v2"], 960, 540)
    assert abs(half.fx - 902.7877 / 2) < 1e-3 and abs(half.cy - 548.9076 / 2) < 1e-3


def test_lidar_scale_and_relief_recover_a_known_mesh():
    from animal3d.lidar import compare_mesh, instance_depth, mask_height_m

    mask = np.zeros((100, 100), bool)
    mask[20:80, 20:80] = True
    # A tilted surface in model units; the "true" animal is that surface at 3 metres per unit.
    ys, xs = np.mgrid[0:100, 0:100]
    mesh = 5 + 0.01 * xs
    lidar = np.zeros((100, 100), np.float32)
    lidar[::5, ::5] = (mesh * 3)[::5, ::5]
    result = compare_mesh(lidar, mask, np.where(mask, mesh, 0))
    assert abs(result["metres_per_unit"] - 3) < 1e-3 and result["relief_error_m"] < 1e-3 and result["lidar_on_mesh"] > 8

    # A flat mesh misses the tilt, which the relief error must show.
    flat = compare_mesh(lidar, mask, np.where(mask, 5.5, 0))
    assert flat["relief_error_m"] > 0.1

    assert np.isnan(instance_depth(np.zeros((100, 100), np.float32), mask)[0])  # no points
    assert abs(mask_height_m(mask, 10.0, 1000.0) - 0.6) < 1e-6  # 60 px at 10 m with f = 1000 px


def test_lidar_points_outside_the_eroded_mask_do_not_count():
    from animal3d.lidar import instance_depth

    mask = np.zeros((100, 100), bool)
    mask[20:80, 20:80] = True
    depth = np.zeros((100, 100), np.float32)
    depth[30:70:3, 30:70:3] = 10.0  # on the animal
    depth[20, 20:80:2] = 90.0  # background leaking along the mask's edge
    assert instance_depth(depth, mask)[0] == 10.0


def test_choose_shape_instances_prefers_side_on_and_holds_tracks_out():
    from animal3d.fit import choose_shape_instances

    track = np.repeat([1, 2, 3, 4, 5], 10)
    usable = np.ones(50, bool)
    side_on = np.arange(50) % 2 == 0  # 25 side-on, below the minimum of 30 here, so all are used
    picks, held = choose_shape_instances(usable, side_on, track)
    assert held == [4] and len(picks) == 40 and not np.isin(track[picks], held).any()
    picks, _ = choose_shape_instances(usable, side_on, track, min_side_on=10)
    assert side_on[picks].all()
    # Too few tracks to spare one.
    assert choose_shape_instances(usable, side_on, np.ones(50, int))[1] == []


def test_make_init_and_torso_is_held_at_unit_length():
    import torch

    from animal3d.body import BODY_BOUNDS, BONE_GROUPS, make_init

    init = make_init(neck=3.0, leg=1.5, thin=0.7)
    assert init["neck"] == (3.0, 0.7) and init["hind_mid"] == (1.5, 0.7) and "torso" not in init
    assert set(BODY_BOUNDS) == set(BONE_GROUPS)
    assert BODY_BOUNDS["torso"][:2] == (1.0, 1.0)  # fixes the size that images cannot determine


def test_gui_rebuilds_a_flattened_folder_upload(tmp_path):
    import cv2

    from app import arrange_upload

    # Browsers deliver rgb/ and depthm/ mixed into one list of files.
    up = tmp_path / "up"
    _write_images(up, ["7.jpg", "8.png"])  # 8.png is an ordinary 8-bit colour image
    depth = up / "7.png"
    cv2.imwrite(str(depth), np.ones((40, 60), np.uint16) * 500)
    stray = up / "9.png"
    cv2.imwrite(str(stray), np.ones((40, 60), np.uint16) * 500)  # depth with no matching colour image
    (up / "metadata.txt").write_text("species: goat\nsource: Zimbabwe\n")
    target = arrange_upload([str(p) for p in up.iterdir()], tmp_path / "out")
    assert sorted(p.name for p in (target / "rgb").iterdir()) == ["7.jpg", "8.png"]
    assert [p.name for p in (target / "depthm").iterdir()] == ["7.png"]
    source = open_source(target)
    assert source.has_lidar and source.species == "goat"


# ------------------------------------------------------------------------------------------ horns


def _smal_body(**kwargs):
    from pathlib import Path

    from animal3d.body import BODY_BOUNDS, AnimalBody
    from run import load_smal

    prima = Path(__file__).resolve().parents[1] / "third_party/PRIMA"
    if not (prima / "data/smal/my_smpl_00781_4_all.pkl").exists():
        pytest.skip("SMAL model not downloaded")
    return AnimalBody(**load_smal(prima), betas=np.zeros(41), bounds=BODY_BOUNDS, **kwargs)


def test_horns_leave_smal_alone_and_stay_on_the_head():
    from animal3d.body import HornShape, Horns

    plain, horned = _smal_body(), _smal_body(horns=HornShape())
    n = plain.n_verts
    assert horned.n_verts == n and horned.faces.max() == n + Horns.n_vertices - 1
    plain_v, plain_j = plain.rest()
    horned_v, horned_j = horned.rest()
    # PRIMA's keypoints index SMAL's vertices, so those must keep their numbers and positions.
    assert torch.allclose(horned_v[:n], plain_v) and torch.allclose(horned_j, plain_j)

    eye = torch.eye(3)
    turn = batch_rodrigues(torch.tensor([[0.0, 0.0, 0.7]]))[0]
    pose = eye.repeat(1, 34, 1, 1).clone()
    still = horned(eye[None], pose)[0]
    pose[0, 15] = turn  # joint 16, the head
    moved = horned(eye[None], pose)[0]
    assert not torch.allclose(moved[n:], still[n:])
    # A rigid turn keeps the distances between horn points.
    assert torch.allclose(torch.pdist(moved[n:]), torch.pdist(still[n:]), atol=1e-4)


def test_horns_are_mirror_images_and_start_near_the_head():
    from animal3d.body import HornShape

    body = _smal_body(horns=HornShape(elevation=40.0, curl_up=60.0, curl_forward=30.0))
    vertices, joints = body.rest()
    n, flip = body.n_verts, torch.tensor([1.0, -1.0, 1.0])
    assert torch.allclose(vertices[n:][body.mirror[n:] - n], vertices[n:] * flip, atol=1e-5)
    assert (vertices[n:n + 80, 1] > 0).float().mean() > 0.9  # the first horn is on +y and leaves outwards
    assert vertices[n:n + 80, 2].max() > vertices[n, 2]  # positive elevation rises
    assert (vertices[n:n + 8].mean(dim=0) - joints[16]).norm() < 0.3


def test_ears_cannot_grow_through_offsets_or_bones():
    body = _smal_body()
    ears = body.weights[:, [33, 34]].sum(dim=1) > 0.99
    with torch.no_grad():
        body.offsets[:] = 0.1
    assert body.symmetric_offsets()[ears].abs().max() < 0.002  # 2% of the free offset at most
    assert body.symmetric_offsets()[~ears].abs().max() > 0.05
    assert body.scales()[body.group_names.index("ears")].max() <= 1.2 + 1e-5


def test_measured_sizes_exclude_horns():
    from animal3d.body import HornShape

    body = _smal_body(horns=HornShape(length=1.2, elevation=90.0))
    assert body.measurements()["horn"] == pytest.approx(1.2)
    assert body.rest()[0][:body.n_verts, 2].max() < body.rest()[0][body.n_verts:, 2].max()


def test_search_keeps_horns_only_when_they_clearly_win(monkeypatch):
    import animal3d.fit as fit

    class Fake:
        def proportions(self):
            return {"neck_to_body": 0.3, "leg_to_body": 0.6}

    for horn_error, expected in ((0.90, False), (0.80, True)):
        made = []

        def make_body(init, horns=None):
            made.append(horns)
            body = Fake()
            body.error = horn_error if horns is not None else 1.0
            return body

        class P:
            def __init__(self, inst): pass
            def to(self, device): return self

        monkeypatch.setattr(fit, "Poses", P)
        monkeypatch.setattr(fit, "optimize", lambda *a, **k: None)
        monkeypatch.setattr(fit, "mask_error", lambda body, poses: body.error)
        inst = type("I", (), {"r0": torch.zeros(1)})()
        _, horns, results, decision = fit.search_body(make_body, inst, log=lambda _: None)
        assert (horns is not None) is expected
        assert any(r["horns"] for r in results) and any(not r["horns"] for r in results)
        assert decision["mask_error_reduction"] == pytest.approx(1 - horn_error)


def _fake_search(monkeypatch, horn_error):
    import animal3d.fit as fit

    class Fake:
        def proportions(self):
            return {"neck_to_body": 0.3, "leg_to_body": 0.6}

    made = []

    def make_body(init, horns=None):
        made.append(horns)
        body = Fake()
        body.error = horn_error if horns is not None else 1.0
        return body

    class P:
        def __init__(self, inst): pass
        def to(self, device): return self

    monkeypatch.setattr(fit, "Poses", P)
    monkeypatch.setattr(fit, "optimize", lambda *a, **k: None)
    monkeypatch.setattr(fit, "mask_error", lambda body, poses: body.error)
    return fit, make_body, made, type("I", (), {"r0": torch.zeros(1)})()


def test_search_builds_the_species_layout_even_when_masks_barely_support_it(monkeypatch):
    fit, make_body, made, inst = _fake_search(monkeypatch, horn_error=0.99)
    _, horns, results, decision = fit.search_body(make_body, inst, horns="nasal_two", log=lambda _: None)
    assert horns is not None and horns.layout == "nasal_two"
    assert all(h is None or h.layout == "nasal_two" for h in made)  # never tries paired horns
    assert decision["mask_error_reduction"] == pytest.approx(0.01)
    assert sum(r["horns"] is None for r in results) == 1  # a single hornless fit, to measure the gain
    # The masks cannot measure these horns, so a typical shape is used, and labelled as such.
    assert horns == fit.TYPICAL_HORNS["nasal_two"][1] and decision["horn_shape"] == "species prior, not measured"

    fit, make_body, made, inst = _fake_search(monkeypatch, horn_error=0.5)
    _, horns, _, decision = fit.search_body(make_body, inst, horns="nasal_two", log=lambda _: None)
    assert horns in fit.horn_starts("nasal_two").values() and decision["horn_shape"] == "fitted to the masks"


def test_search_without_horns_never_builds_any(monkeypatch):
    fit, make_body, made, inst = _fake_search(monkeypatch, horn_error=0.5)
    _, horns, _, decision = fit.search_body(make_body, inst, horns="none", log=lambda _: None)
    assert horns is None and all(h is None for h in made) and decision["layout"] is None


def test_nose_horns_stand_on_the_snout_midline_front_one_largest():
    from animal3d.body import HornShape

    body = _smal_body(horns=HornShape(layout="nasal_two", length=0.25, radius=0.05, elevation=75.0, rear_length=0.4))
    vertices, joints = body.rest()
    n, flip = body.n_verts, torch.tensor([1.0, -1.0, 1.0])
    horn = vertices[n:]
    assert torch.allclose(horn[body.mirror[n:] - n], horn * flip, atol=1e-5)  # each is its own mirror image
    front, rear = horn[:80], horn[80:]
    snout = joints[32] - joints[16]
    along = lambda p: float((p - joints[16]) @ snout / snout.norm() ** 2)
    assert 1.3 < along(front[:8].mean(0)) < 2.1 and along(rear[:8].mean(0)) < along(front[:8].mean(0))
    assert front[:, 2].max() > vertices[:n, 2][((vertices[:n] - joints[32]).norm(dim=1) < 0.2)].max()  # rises above the nose
    height = lambda tube: float((tube[-8:].mean(0) - tube[:8].mean(0)).norm())
    assert height(front) > 2 * height(rear)
    assert body.measurements()["rear_horn"] == pytest.approx(0.1)


def test_single_nose_horn_has_no_rear_horn():
    from animal3d.body import HornShape

    body = _smal_body(horns=HornShape(layout="nasal_one", rear_length=0.5))
    assert body.measurements()["rear_horn"] == 0.0
    rear = body.rest()[0][body.n_verts + 80:]
    assert float((rear - rear.mean(0)).norm(dim=1).max()) < 1e-4  # collapsed inside the head


def test_species_labels_and_classifier_choose_the_horn_layout():
    from animal3d.species import decide

    assert decide("Ankle_cow", None).layout == "paired"
    assert decide("camel", None).layout == "none"
    assert decide("giraffe", None).layout == "none"  # ossicones are not modelled
    assert decide("Indian rhino", None).layout == "nasal_one"
    assert decide("Rhino", None).layout is None  # a group: needs the classifier
    rhino_votes = {"black_rhino": 0.31, "white_rhino": 0.24, "hippo": 0.28, "indian_rhino": 0.05}
    assert decide("Rhino", rhino_votes).layout == "nasal_two"
    unsure = decide(None, rhino_votes | {"horse": 0.12})
    assert unsure.layout == "nasal_two"  # 0.55 of all votes are on two-horned rhinos
    assert decide(None, {"black_rhino": 0.4, "hippo": 0.3, "goat": 0.3}).layout is None  # no layout has half
    assert decide(None, {"black_rhino": 0.3, "hippo": 0.35, "horse": 0.35}).layout == "none"
    assert decide(None, None).layout is None


# ------------------------------------------------------------------------------- humps and LiDAR fit


def test_humps_are_two_separate_symmetric_domes_on_the_back():
    from animal3d.body import Humps

    plain, humped = _smal_body(), _smal_body(humps=2)
    n = plain.n_verts
    with torch.no_grad():
        humped.humps.raw[:, 1] = 0.2
    a, ja = plain.rest()
    b, jb = humped.rest()
    assert torch.allclose(b[:n], a) and torch.allclose(jb, ja)  # SMAL's vertices untouched
    assert len(b) == n + 2 * Humps.per_hump and humped.faces.max() == len(b) - 1
    domes = b[n:].reshape(2, Humps.per_hump, 3)
    tops = domes[:, 0]
    assert (tops[:, 2] > 0.31 + 0.15).all()  # the template's back is at most 0.31 high between hip and shoulder
    # Two humps, one behind the other, with the back dipping between them: the rear dome's front edge does not
    # reach the front dome's rear edge at skin height.
    rear, front = (tops[:, 0] - ja[1, 0]) / (ja[6, 0] - ja[1, 0])
    assert rear < 0.45 < 0.55 < front
    flip = torch.tensor([1.0, -1.0, 1.0])
    assert torch.allclose(b[n:][humped.mirror[n:] - n], b[n:] * flip, atol=1e-5)
    # Outward-facing triangles: the normals point away from each dome's axis.
    tri = torch.as_tensor(humped.faces[len(plain.faces):])
    v = b
    normal = torch.linalg.cross(v[tri[:, 1]] - v[tri[:, 0]], v[tri[:, 2]] - v[tri[:, 0]])
    centre = v[tri].mean(dim=1)
    axis_point = torch.where((tri[:, :1] - n) < Humps.per_hump, tops[0], tops[1]).clone()
    axis_point = axis_point - torch.tensor([0.0, 0.0, 0.1])
    assert ((normal * (centre - axis_point)).sum(dim=1) > 0).float().mean() > 0.95


def test_humps_move_with_the_back_and_report_their_height():
    body = _smal_body(humps=1)
    n = body.n_verts
    skin = body.skin()[n:]
    assert torch.allclose(skin.sum(dim=1), torch.ones(len(skin)))
    assert skin[:, :7].sum(dim=1).min() > 0.9  # torso joints only
    eye = torch.eye(3)
    pose = eye.repeat(1, 34, 1, 1).clone()
    still = body(eye[None], pose)[0]
    pose[0, 15] = batch_rodrigues(torch.tensor([[0.0, 0.0, 0.7]]))[0]  # turning the head leaves the hump alone
    assert torch.allclose(body(eye[None], pose)[0][n:], still[n:], atol=1e-5)
    assert "hump1_height" in body.measurements()


def test_species_gives_camels_their_humps():
    from animal3d.species import decide

    assert decide("Bactrian camel", None).humps == 2 and decide("dromedary", None).humps == 1
    group = decide("camel", None)
    assert group.layout == "none" and group.humps == 0  # how many humps needs the classifier
    assert decide("camel", {"bactrian_camel": 0.36, "dromedary": 0.24, "horse": 0.26}).humps == 2
    assert decide("giraffe", None).humps == 0


def test_species_gives_camels_small_ears_and_others_fitted_ones():
    from animal3d.species import decide

    assert decide("Bactrian camel", None).ears == 0.45
    assert decide("camel", None).ears == 0.45  # both camels agree, so no classifier is needed
    assert decide("camel", {"bactrian_camel": 0.36, "dromedary": 0.24, "horse": 0.26}).ears == 0.45
    assert decide("giraffe", None).ears is None and decide("rhino", None).ears is None


def test_depth_loss_sees_which_way_the_body_turns():
    from animal3d.fit import depth_loss

    # A plane of vertices facing the camera, tilted so its right side is further away.
    gx, gy = torch.meshgrid(torch.linspace(-1, 1, 40), torch.linspace(-1, 1, 40), indexing="xy")
    plane = lambda tilt: torch.stack([gx.flatten(), gy.flatten(), tilt * gx.flatten()], dim=1)[None]
    cam_t = torch.tensor([[0.0, 0.0, 10.0]])
    uv = lambda v: (v[..., :2] / (v[..., 2:] + 10.0)) * 2 + 0.5  # a toy projection into the crop
    picks = torch.arange(0, 1600, 37)
    truth = plane(0.5)
    lidar_uv, lidar_d = uv(truth)[:, picks], (truth[:, picks, 2] + 10.0) * 1.7  # metres, 1.7 per unit
    ok = torch.ones_like(lidar_d, dtype=torch.bool)
    same = depth_loss(truth, cam_t, uv(truth), lidar_uv, lidar_d, ok)
    mirrored = depth_loss(plane(-0.5), cam_t, uv(plane(-0.5)), lidar_uv, lidar_d, ok)
    flat = depth_loss(plane(0.0), cam_t, uv(plane(0.0)), lidar_uv, lidar_d, ok)
    assert same < 1e-4 < flat < mirrored
    none = depth_loss(truth, cam_t, uv(truth), lidar_uv, lidar_d, torch.zeros_like(ok))
    assert none == 0


def test_fit_and_check_use_separate_lidar_points():
    from animal3d.lidar import compare_mesh, fitting_points, split_pattern

    depth = np.full((64, 64), 10.0, np.float32)
    mask = np.ones((64, 64), bool)
    xy, d = fitting_points(depth, mask, 10_000, np.random.default_rng(0))
    pattern = split_pattern(64, 64)
    assert len(d) and pattern[xy[:, 1].astype(int), xy[:, 0].astype(int)].all()
    assert 0.3 < pattern.mean() < 0.7
    depth[:, 48:] = 30.0  # past the outline: dropped as stray returns
    _, d = fitting_points(depth, mask, 10_000, np.random.default_rng(0))
    assert len(d) and d.max() < 12.0
    held = compare_mesh(np.full((64, 64), 10.0, np.float32), mask, np.full((64, 64), 5.0), where=~pattern)
    assert held["lidar_on_mesh"] == int((~pattern).sum()) and held["flat_relief_m"] == 0.0


def test_depth_loss_cannot_be_lowered_by_uncovering_points():
    from animal3d.fit import depth_loss

    gx, gy = torch.meshgrid(torch.linspace(-1, 1, 40), torch.linspace(-1, 1, 40), indexing="xy")
    full = torch.stack([gx.flatten(), gy.flatten(), 0.5 * gx.flatten()], dim=1)[None]
    cam_t = torch.tensor([[0.0, 0.0, 10.0]])
    uv = lambda v: (v[..., :2] / (v[..., 2:] + 10.0)) * 2 + 0.5
    picks = torch.arange(0, 1600, 37)
    lidar_uv = uv(full)[:, picks]
    lidar_d = (full[:, picks, 2] + 10.0) * 1.7 + 0.05 * torch.randn(1, len(picks), generator=torch.Generator().manual_seed(0))
    ok = torch.ones_like(lidar_d, dtype=torch.bool)
    half = full[:, gx.flatten() < 0]  # the same surface, missing its right half
    assert depth_loss(full, cam_t, uv(full), lidar_uv, lidar_d, ok) < depth_loss(half, cam_t, uv(half), lidar_uv, lidar_d, ok)


def test_depth_loss_does_not_push_the_mesh_along_the_view_axis():
    from animal3d.fit import depth_loss

    gx, gy = torch.meshgrid(torch.linspace(-1, 1, 40), torch.linspace(-1, 1, 40), indexing="xy")
    surface = torch.stack([gx.flatten(), gy.flatten(), 0.3 * gx.flatten() ** 2], dim=1)[None]
    uv = lambda v: (v[..., :2] / (v[..., 2:] + 10.0)) * 2 + 0.5
    picks = torch.arange(0, 1600, 37)
    noise = torch.randn(1, len(picks), generator=torch.Generator().manual_seed(1))
    lidar_d = (surface[:, picks, 2] + 10.0) * 1.7 + 0.3 * noise.abs()  # skewed: mean and median differ
    ok = torch.ones_like(lidar_d, dtype=torch.bool)
    cam_t = torch.tensor([[0.0, 0.0, 10.0]], requires_grad=True)
    depth_loss(surface, cam_t, uv(surface).detach(), uv(surface)[:, picks], lidar_d, ok).backward()
    assert abs(float(cam_t.grad[0, 2])) < 1e-6


# ------------------------------------------------------------------------------- kinematics


def test_joint_angles_straight_and_bent_legs_and_swing_sign():
    from animal3d.kinematics import joint_angles

    joints = np.zeros((2, 35, 3))
    joints[:, 1], joints[:, 6] = [-0.5, 0, 0.3], [0.2, 0, 0.3]  # hip, shoulder: facing +x
    for top, mid, low, foot in ((7, 8, 9, 10), (11, 12, 13, 14), (17, 18, 19, 20), (21, 22, 23, 24)):
        joints[:, top] = [0, 0, 0.1]
        joints[:, mid] = [0, 0, -0.1]
        joints[:, low] = [0, 0, -0.3]
        joints[:, foot] = [0, 0, -0.5]
    joints[1, 19] = [0.2, 0, -0.1]  # frame 1: hind hock folded forward at the stifle
    joints[1, 20] = [0.2, 0, -0.3]
    table = joint_angles(joints)
    assert table.left_stifle[0] == pytest.approx(180.0) and table.left_stifle[1] == pytest.approx(90.0)
    assert table.left_fore[0] == pytest.approx(0.0, abs=1e-6)
    assert table.left_hind[1] > 0  # foot ahead of the hip: swung forwards


def test_stride_frequency_finds_a_walking_rhythm():
    from animal3d.kinematics import stride_frequency

    t = np.arange(100) / 10.0
    freq, share = stride_frequency(20 * np.sin(2 * np.pi * 0.6 * t) + np.random.default_rng(0).normal(0, 3, 100), 10.0)
    assert freq == pytest.approx(0.6, abs=0.05) and share > 0.5
