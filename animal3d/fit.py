"""Measures an animal's body from its instance masks, with no knowledge of its species.

Uses a bidirectional distance loss on projected vertices rather than a soft rasterizer. Rasterizer
gradients are local, so they fade when a mesh part lies far outside the mask, which is exactly the
state of a neck that needs to change by a large amount. Even so, descent from one start can settle in
the wrong body plan, so `search_body` tries several and keeps the one that the masks support best.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from smplx.lbs import batch_rodrigues

from animal3d.body import HornShape, make_init

CROP_RES = 128
MASK_SAMPLES = 400
CROP_PAD = 1.6
NECK_HEAD_JOINTS = (15, 16)  # 1-based over the root, i.e. indices 14, 15 of the pose array

# Starting body plans for the search, as multiples of the torso length.
SEARCH_NECK = (1.0, 2.0, 3.2)
SEARCH_LEG = (1.0, 1.5)
SEARCH_THIN = (1.0, 0.7)
# Starting horns. Each only has to land near the right kind of horn; the fit refines it.
HORN_STARTS = {
    "out and up": HornShape(length=0.55, radius=0.05, elevation=10.0, curl_up=70.0),
    "upright": HornShape(length=0.35, radius=0.05, elevation=65.0),
    "swept back": HornShape(length=0.5, radius=0.04, elevation=20.0, sweep=-50.0),
    "short": HornShape(length=0.12, radius=0.035, elevation=70.0),
}
# Nose horns (rhinos), used only when the species says so: masks rarely show them, as they cut the horn off.
NASAL_STARTS = {
    "tall": HornShape(layout="nasal_two", length=0.22, radius=0.05, elevation=75.0, curl_up=20.0, nose_position=1.75, rear_length=0.4),
    "low": HornShape(layout="nasal_two", length=0.12, radius=0.045, elevation=55.0, curl_up=10.0, nose_position=1.8, rear_length=0.4),
}
HORN_LAYOUTS = ("paired", "nasal_two", "nasal_one")
# Used when the species has horns but the masks cannot measure them. A rhino's front horn is often about 0.6 m on a
# head of about 0.7 m, and the rear one a third to a half of it (sources in animal3d/species.py). Paired horns vary too
# much between species for one typical shape, so they keep the start that fitted best.
TYPICAL_HORNS = {
    "nasal_two": ("typical rhino", HornShape(layout="nasal_two", length=0.24, radius=0.05, elevation=75.0, curl_up=25.0,
                                            nose_position=1.8, rear_length=0.4, rear_gap=0.45)),
    "nasal_one": ("typical one-horned rhino", HornShape(layout="nasal_one", length=0.16, radius=0.05, elevation=75.0,
                                                        curl_up=20.0, nose_position=1.8)),
}


def horn_starts(layout: str) -> dict[str, HornShape]:
    """Starting horns for a layout ("paired", "nasal_two" or "nasal_one")."""
    if layout == "paired":
        return HORN_STARTS
    starts = {name: replace(shape, layout=layout) for name, shape in NASAL_STARTS.items()}
    if layout == "nasal_one":
        starts = {name: replace(shape, rear_length=0.0) for name, shape in starts.items()}
    return starts
# More parameters always fit a little better, and a horn can fill a gap left by a poorly fitted head. So horns must
# clearly reduce the mask error. Set from few clips: Ankole cattle gained 17% and 52%; a giraffe 9% and a rhino 1%.
# Animals with short horns gain little, so they may well be fitted without any: a missing horn, not a false one.
HORN_MARGIN = 0.13


def choose_shape_instances(usable: np.ndarray, side_on: np.ndarray, track: np.ndarray, min_side_on: int = 30,
                           cap: int = 600) -> tuple[np.ndarray, list[int]]:
    """Which instances to measure the body on, and which tracks to keep back to test it.

    Side-on views carry the proportions, so prefer them when there are enough. One track in four is held
    out, given at least four, to check the shape on animals that it did not come from.
    """
    pool = usable & side_on if (usable & side_on).sum() >= min_side_on else usable
    tracks = sorted(set(track[pool].tolist()))
    held_out = tracks[3::4] if len(tracks) >= 4 else []
    picks = np.where(pool & ~np.isin(track, held_out))[0]
    if len(picks) > cap:
        picks = picks[np.linspace(0, len(picks) - 1, cap).astype(int)]
    return picks, held_out


def prepare_mask(mask: np.ndarray, box: np.ndarray, rng: np.random.Generator):
    """Crop centre and size, plus a distance map and point samples, in crop-normalised units (0..1)."""
    center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=np.float32)
    size = float(max(box[2] - box[0], box[3] - box[1]) * CROP_PAD)
    scale = CROP_RES / size
    matrix = np.array([[scale, 0, CROP_RES / 2 - center[0] * scale], [0, scale, CROP_RES / 2 - center[1] * scale]], np.float32)
    crop = cv2.warpAffine(mask.astype(np.uint8), matrix, (CROP_RES, CROP_RES), flags=cv2.INTER_NEAREST)
    if not crop.any():  # a mask too thin to survive resampling still needs a target
        crop[CROP_RES // 2, CROP_RES // 2] = 1
    distance = cv2.distanceTransform((crop == 0).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32) / CROP_RES

    # Half the samples on the outline, which carries the shape; half anywhere in the mask.
    outline = crop - cv2.erode(crop, np.ones((3, 3), np.uint8))
    outline_pts, area_pts = np.argwhere(outline > 0), np.argwhere(crop > 0)
    picks = [
        pts[rng.integers(0, len(pts), MASK_SAMPLES // 2)] for pts in (outline_pts, area_pts)
    ]
    samples = (np.concatenate(picks)[:, ::-1].astype(np.float32) + 0.5) / CROP_RES
    return center, size, distance, samples


@dataclass
class Instances:
    """Masks, cameras and PRIMA's starting estimates for each animal instance, all as tensors."""

    r0: torch.Tensor  # (N, 3, 3)
    p0: torch.Tensor  # (N, 34, 3, 3)
    t0: torch.Tensor  # (N, 3)
    camera: torch.Tensor  # (N, 4) fx, fy, cx, cy
    center: torch.Tensor  # (N, 2)
    size: torch.Tensor  # (N,)
    distance: torch.Tensor  # (N, 1, R, R)
    samples: torch.Tensor  # (N, K, 2)
    track: np.ndarray
    frame: np.ndarray
    # LiDAR points the fit may use, in crop coordinates like `samples`, with depths in metres; `lidar_ok` marks real
    # points in the padded arrays. None when there is no LiDAR.
    lidar_uv: torch.Tensor | None = None  # (N, L, 2)
    lidar_d: torch.Tensor | None = None  # (N, L)
    lidar_ok: torch.Tensor | None = None  # (N, L) bool

    def __len__(self) -> int:
        return len(self.r0)

    def subset(self, idx: np.ndarray) -> "Instances":
        t = torch.as_tensor(idx, device=self.r0.device)
        pick = lambda x: None if x is None else x[t]
        return Instances(self.r0[t], self.p0[t], self.t0[t], self.camera[t], self.center[t], self.size[t],
                         self.distance[t], self.samples[t], self.track[idx], self.frame[idx],
                         pick(self.lidar_uv), pick(self.lidar_d), pick(self.lidar_ok))


class Poses(torch.nn.Module):
    """Per-instance corrections on top of PRIMA's estimate, so that it remains the starting point."""

    def __init__(self, inst: Instances):
        super().__init__()
        self.inst = inst
        n = len(inst)
        device = inst.r0.device
        self.d_orient = torch.nn.Parameter(torch.zeros(n, 3, device=device))
        self.d_pose = torch.nn.Parameter(torch.zeros(n, 34, 3, device=device))
        self.d_cam = torch.nn.Parameter(torch.zeros(n, 3, device=device))  # relative to depth

    def matrices(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        inst = self.inst
        orient = batch_rodrigues(self.d_orient[idx]) @ inst.r0[idx]
        n = len(orient)
        delta = batch_rodrigues(self.d_pose[idx].reshape(-1, 3)).reshape(n, 34, 3, 3)
        return orient, delta @ inst.p0[idx]

    def translation(self, idx) -> torch.Tensor:
        t0 = self.inst.t0[idx]
        return t0 + self.d_cam[idx] * t0[:, 2:]


def project_to_crop(vertices, cam_t, camera, center, size) -> torch.Tensor:
    """Vertices (N, V, 3) to crop-normalised coordinates, with 0..1 inside the crop. `camera`: (N, 4)."""
    points = vertices + cam_t[:, None]
    pixels = camera[:, None, :2] * points[..., :2] / points[..., 2:].clamp(min=1e-3) + camera[:, None, 2:]
    return (pixels - center[:, None]) / size[:, None, None] + 0.5


def mask_loss(uv: torch.Tensor, distance: torch.Tensor, samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(Mesh outside the mask, mask not covered by the mesh), as mean squared distances."""
    inside = F.grid_sample(distance, (uv * 2 - 1)[:, None], align_corners=False, padding_mode="border")[:, 0, 0]
    beyond_crop = (uv - 0.5).abs().sub(0.5).clamp(min=0).sum(dim=-1)
    squared = (inside + beyond_crop).pow(2)
    # The mean alone hides a few runaway limb vertices among thousands, so add the worst 5% per animal.
    worst = squared.topk(max(squared.shape[1] // 20, 1), dim=1).values.mean()
    outside = squared.mean() + worst
    coverage = torch.cdist(samples, uv).min(dim=2).values.pow(2).mean()
    return outside, coverage


DEPTH_RADIUS = 2.0 / CROP_RES  # a LiDAR point takes the depth of the nearest-to-camera vertex within this
DEPTH_HUBER = 0.03  # model units (about 5 cm): beyond it, a residual counts linearly, so stray returns weigh little
MIN_DEPTH_POINTS = 8
# What a LiDAR point off the mesh costs: the Huber loss of a 0.1-unit (about 15 cm) residual.
UNCOVERED_COST = DEPTH_HUBER * (0.1 - DEPTH_HUBER / 2)


def depth_loss(vertices: torch.Tensor, cam_t: torch.Tensor, uv: torch.Tensor, lidar_uv, lidar_d, lidar_ok) -> torch.Tensor:
    """How far the mesh's depth relief departs from the LiDAR's, in model units.

    Relief is depth about the animal's middle, as in the LiDAR check: overall distance and scale are left to the
    masks and to LiDAR scaling afterwards. This is what outlines cannot tell, such as which way the body turns in depth.
    The mesh's depth at a LiDAR point is that of the front-most vertex projecting next to it: no rasteriser needed.
    """
    z = (vertices + cam_t[:, None])[..., 2]
    near = torch.cdist(lidar_uv, uv) < DEPTH_RADIUS  # (B, L, V)
    inf = torch.tensor(float("inf"), device=z.device)
    front = torch.where(near, z[:, None, :], inf).min(dim=2).values  # (B, L)
    covered = lidar_ok & torch.isfinite(front)
    enough = (lidar_ok.sum(dim=1) >= MIN_DEPTH_POINTS) & (covered.sum(dim=1) >= MIN_DEPTH_POINTS)
    if not enough.any():
        return torch.zeros((), device=z.device)
    valid = covered & enough[:, None]
    nan = torch.tensor(float("nan"), device=z.device)
    lidar = torch.where(valid, lidar_d, nan)
    mesh = torch.where(valid, front, nan)
    # The mesh's median keeps its gradient: then moving the whole mesh nearer or further changes nothing. Detached, it
    # let every point push the mesh along the view axis without end, against the masks (sizes swung 65% per frame).
    mesh_mid = mesh.nanmedian(dim=1).values
    with torch.no_grad():
        lidar_mid = lidar.nanmedian(dim=1).values
        metres_per_unit = (lidar_mid / mesh_mid).clamp(min=1e-6)
    residual = (lidar - lidar_mid[:, None]) / metres_per_unit[:, None] - (mesh - mesh_mid[:, None])
    residual = torch.where(valid, residual, torch.zeros_like(residual))
    per_point = F.huber_loss(residual, torch.zeros_like(residual), reduction="none", delta=DEPTH_HUBER)
    # A LiDAR point the mesh does not cover still costs something. Otherwise the cheapest fix for an awkward point is
    # to shrink or turn the mesh away from it, which is what happened: Rhino_3's IoU fell from 0.88 to 0.67.
    uncovered = (lidar_ok & ~covered).sum(dim=1).float() * UNCOVERED_COST
    total = (per_point * valid).sum(dim=1) + uncovered
    return (total / lidar_ok.sum(dim=1).clamp(min=1))[enough].mean()


def pose_prior(poses: Poses, idx) -> torch.Tensor:
    """Stay near PRIMA's pose, but let the neck and head move: its neck rotations suit its neck length."""
    weights = torch.ones(34, device=poses.d_pose.device)
    for joint in NECK_HEAD_JOINTS:
        weights[joint - 1] = 0.05
    return (poses.d_pose[idx].pow(2).sum(dim=-1) * weights).mean()


def temporal_loss(poses: Poses) -> torch.Tensor:
    """Damps frame-to-frame jitter in the corrections within each track."""
    inst = poses.inst
    if not hasattr(inst, "_pairs"):
        pairs, last = [], {}
        for i in np.lexsort((inst.frame, inst.track)):
            key = int(inst.track[i])
            if key in last and inst.frame[i] - inst.frame[last[key]] <= 2:
                pairs.append((last[key], i))
            last[key] = i
        inst._pairs = torch.as_tensor(np.array(pairs).reshape(-1, 2), device=poses.d_pose.device)
    a, b = inst._pairs[:, 0], inst._pairs[:, 1]
    if len(a) == 0:
        return torch.zeros((), device=poses.d_pose.device)
    return (poses.d_pose[a] - poses.d_pose[b]).pow(2).mean() + (poses.d_orient[a] - poses.d_orient[b]).pow(2).mean()


class IndexedAdam:
    """Adam over per-instance parameters, updating only the instances in the current batch.

    Standard Adam keeps moving unsampled instances on their stale momentum, which sends them adrift.
    """

    def __init__(self, params_and_lrs, betas=(0.9, 0.999), eps=1e-8):
        self.items = [
            {"param": p, "lr": lr, "m": torch.zeros_like(p), "v": torch.zeros_like(p), "t": torch.zeros(len(p), device=p.device)}
            for p, lr in params_and_lrs
        ]
        self.b1, self.b2, self.eps = betas[0], betas[1], eps

    def zero_grad(self) -> None:
        for item in self.items:
            item["param"].grad = None

    @torch.no_grad()
    def step(self, idx: torch.Tensor) -> None:
        idx = idx.unique()
        for item in self.items:
            grad = item["param"].grad[idx]
            item["t"][idx] += 1
            item["m"][idx] = self.b1 * item["m"][idx] + (1 - self.b1) * grad
            item["v"][idx] = self.b2 * item["v"][idx] + (1 - self.b2) * grad.pow(2)
            view = (-1,) + (1,) * (grad.dim() - 1)
            t = item["t"][idx].view(view)
            m_hat = item["m"][idx] / (1 - self.b1**t)
            v_hat = item["v"][idx] / (1 - self.b2**t)
            item["param"][idx] -= item["lr"] * m_hat / (v_hat.sqrt() + self.eps)


def optimize(body, poses: Poses, iters, batch, lr_pose, lr_shape=0.0, lr_offsets=0.0,
             w_pose=1e-2, w_temporal=0.0, w_laplacian=0.0, w_lateral=0.0, w_scale_prior=1e-3, w_depth=0.0, seed=0, log=print,
             lr_betas=0.0, w_betas=1e-4, w_orient=0.0) -> None:
    """Parameters with a zero learning rate stay fixed. Betas stay at their starting value throughout:
    left free they trade off against bone scales and drift far outside SMAL's plausible range. `lr_betas` frees them
    (with a pull of `w_betas` towards the start) only for the SMAL-shape-space baseline of the ablation.
    `w_orient` keeps the body's orientation near its starting value (test-time refinement of a network's prediction)."""
    inst = poses.inst
    # Camera corrections are relative to depth, so the same step moves them far more in pixels.
    pose_optimizer = IndexedAdam([(poses.d_orient, lr_pose), (poses.d_pose, lr_pose), (poses.d_cam, lr_pose * 0.1)])
    shape_groups = []
    if lr_shape:
        shape_groups.append({"params": [body.log_scale], "lr": lr_shape})
        if body.horns is not None and body.horns.raw.requires_grad:  # a species-prior horn is held fixed
            shape_groups.append({"params": [body.horns.raw], "lr": lr_shape})
        if body.humps is not None:
            shape_groups.append({"params": [body.humps.raw], "lr": lr_shape})
    if lr_offsets:
        shape_groups.append({"params": [body.offsets], "lr": lr_offsets})
    if lr_betas:
        shape_groups.append({"params": [body.betas], "lr": lr_betas})
    shape_optimizer = torch.optim.Adam(shape_groups) if shape_groups else None
    generator = torch.Generator(device=inst.r0.device).manual_seed(seed)
    scale_start = body.log_scale.detach().clone()
    betas_start = body.betas.detach().clone()

    for step in range(iters):
        idx = torch.randperm(len(inst), generator=generator, device=inst.r0.device)[:batch]
        orient, pose = poses.matrices(idx)
        vertices = body(orient, pose)
        uv = project_to_crop(vertices, poses.translation(idx), inst.camera[idx], inst.center[idx], inst.size[idx])
        outside, coverage = mask_loss(uv, inst.distance[idx], inst.samples[idx])

        loss = outside + coverage + w_pose * pose_prior(poses, idx)
        if w_orient:
            loss = loss + w_orient * poses.d_orient[idx].pow(2).sum(dim=-1).mean()
        depth = None
        if w_depth and inst.lidar_d is not None:
            depth = depth_loss(vertices, poses.translation(idx), uv, inst.lidar_uv[idx], inst.lidar_d[idx], inst.lidar_ok[idx])
            loss = loss + w_depth * depth
        if w_temporal:
            loss = loss + w_temporal * temporal_loss(poses)
        if lr_shape:
            loss = loss + w_scale_prior * (body.log_scale - scale_start).pow(2).mean()
        if lr_betas:
            loss = loss + w_betas * (body.betas - betas_start).pow(2).sum()
        if lr_offsets:
            loss = loss + w_laplacian * body.laplacian_loss()
            # Mostly side-on footage says little about width, so width offsets must earn their keep.
            loss = loss + w_lateral * body.symmetric_offsets()[:, 1].pow(2).mean()

        pose_optimizer.zero_grad()
        if shape_optimizer:
            shape_optimizer.zero_grad()
        loss.backward()
        pose_optimizer.step(idx)
        if shape_optimizer:
            shape_optimizer.step()
        if log and (step % max(iters // 5, 1) == 0 or step == iters - 1):
            extra = f" depth {depth.item():.5f}" if depth is not None else ""
            log(f"  step {step:5d} outside {outside.item():.5f} coverage {coverage.item():.5f}{extra} {body.proportions()}")


@torch.no_grad()
def mask_error(body, poses: Poses, chunk: int = 64) -> float:
    """Mean of both mask-loss terms over every instance, for comparing fits."""
    inst = poses.inst
    total = 0.0
    for start in range(0, len(inst), chunk):
        idx = torch.arange(start, min(start + chunk, len(inst)), device=inst.r0.device)
        uv = project_to_crop(body(*poses.matrices(idx)), poses.translation(idx), inst.camera[idx], inst.center[idx], inst.size[idx])
        outside, coverage = mask_loss(uv, inst.distance[idx], inst.samples[idx])
        total += float(outside + coverage) * len(idx)
    return total / len(inst)


def search_body(make_body, inst: Instances, iters: int = 300, batch: int = 32, horns: str = "masks",
                log=print) -> tuple[dict, HornShape | None, list[dict], dict]:
    """Tries starting body plans; returns the best plan, its horns (or None), all results, and the horn decision.

    `horns` is "none", "masks" (paired horns, kept only if the masks clearly support them), or a layout known from the
    species ("paired", "nasal_two", "nasal_one"), which is then always built: the masks only shape it.

    `make_body(init, horns)` builds a fresh body. Giving each a short fit before comparing matters: a long-neck start
    only fits well once its pose has settled, and comparing the raw starts would just favour PRIMA's own shape.
    """

    def trial(init: dict, shape: HornShape | None):
        body = make_body(init, shape)
        poses = Poses(inst).to(inst.r0.device)
        optimize(body, poses, iters, batch, lr_pose=0.01, log=None)
        optimize(body, poses, iters, batch, lr_pose=0.01, lr_shape=0.005, log=None)
        return mask_error(body, poses), body

    known = horns in HORN_LAYOUTS
    horn_name, horn_start = None, None
    if horns != "none":
        starts = horn_starts(horns if known else "paired")
        errors = {name: trial(make_init(), start)[0] for name, start in starts.items()}
        log("  horn starts on the default body: " + ", ".join(f"{k} {v:.5f}" for k, v in errors.items()))
        horn_name = min(errors, key=errors.get)
        horn_start = starts[horn_name]

    # With a known layout, plans are compared with horns on; one hornless fit of the best plan measures their gain.
    variants = [(horn_name, horn_start)] if known else [(None, None)] + ([(horn_name, horn_start)] if horns == "masks" else [])
    results = []
    for neck in SEARCH_NECK:
        for leg in SEARCH_LEG:
            for thin in SEARCH_THIN:
                init = make_init(neck, leg, thin)
                for name, start in variants:
                    error, body = trial(init, start)
                    results.append({"neck": neck, "leg": leg, "thin": thin, "horns": name, "error": error, **body.proportions()})
                    log(f"  start neck {neck} leg {leg} thin {thin} horns {name}: error {error:.5f}, neck/body {results[-1]['neck_to_body']:.2f}")
    measured = True
    if known:
        best = min(results, key=lambda r: r["error"])
        error, body = trial(make_init(best["neck"], best["leg"], best["thin"]), None)
        results.append({"neck": best["neck"], "leg": best["leg"], "thin": best["thin"], "horns": None, "error": error, **body.proportions()})
        log(f"  same plan without horns: error {error:.5f}")
        plain, horned = results[-1], best
        if horned["error"] >= plain["error"] * (1 - HORN_MARGIN):
            # The masks do not show these horns (rhino masks cut them off), and fitting one anyway turns it into a
            # blob that pads the outline. Build a typical horn of the layout instead, held fixed and labelled as such.
            measured = False
            horn_name, horn_start = TYPICAL_HORNS.get(horns, (horn_name, horn_start))
            log(f"  horns gain {1 - horned['error'] / plain['error']:.1%} < {HORN_MARGIN:.0%}: using a typical '{horn_name}' horn, not fitted")
    else:
        plain = min((r for r in results if r["horns"] is None), key=lambda r: r["error"])
        horned = min((r for r in results if r["horns"] is not None), key=lambda r: r["error"], default=None)
        best = horned if horned is not None and horned["error"] < plain["error"] * (1 - HORN_MARGIN) else plain
    decision = {"mode": horns, "layout": horn_start.layout if best["horns"] else None, "start": horn_name if best["horns"] else None,
                "horn_shape": None if not best["horns"] else ("fitted to the masks" if measured else "species prior, not measured")}
    if horned is not None:
        decision["mask_error_reduction"] = 1.0 - horned["error"] / plain["error"]
    return make_init(best["neck"], best["leg"], best["thin"]), (horn_start if best["horns"] else None), results, decision


def fit_body(body, inst: Instances, bone_iters: int, offset_iters: int, batch: int, w_depth: float = 0.0, log=print) -> None:
    """Refines the chosen start into the final shape: settle pose, then bone scales, then surface detail."""
    poses = Poses(inst).to(inst.r0.device)
    log("Settling pose on the masks" + (" and LiDAR" if w_depth and inst.lidar_d is not None else ""))
    optimize(body, poses, 1500, batch, lr_pose=0.01, w_depth=w_depth, log=log)
    log("Fitting bone lengths and thicknesses")
    optimize(body, poses, bone_iters, batch, lr_pose=0.01, lr_shape=0.003, w_depth=w_depth, log=log)
    log("Fitting surface offsets")
    optimize(body, poses, offset_iters, batch, lr_pose=0.005, lr_shape=0.001, lr_offsets=0.0005,
             w_laplacian=40.0, w_lateral=0.5, w_depth=w_depth, log=log)
    for parameter in body.parameters():
        parameter.requires_grad_(False)


@torch.no_grad()
def posed_vertices(body, poses: Poses, chunk: int = 64) -> np.ndarray:
    parts = []
    for start in range(0, len(poses.inst), chunk):
        idx = torch.arange(start, min(start + chunk, len(poses.inst)), device=poses.inst.r0.device)
        parts.append(body(*poses.matrices(idx)).cpu().numpy())
    return np.concatenate(parts)
