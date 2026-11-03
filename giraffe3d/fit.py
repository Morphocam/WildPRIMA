"""Fits the giraffe body and each animal's pose to instance masks.

Uses a bidirectional distance loss on projected vertices rather than a soft rasterizer. Rasterizer
gradients are local, so they fade when a mesh part lies far outside the mask, which is exactly the
state of a neck that needs to change by a large amount.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from smplx.lbs import batch_rodrigues

CROP_RES = 128
MASK_SAMPLES = 400
CROP_PAD = 1.6
NECK_HEAD_JOINTS = (15, 16)  # 1-based over the root, i.e. indices 14, 15 of the pose array


def prepare_mask(mask: np.ndarray, box: np.ndarray, rng: np.random.Generator):
    """Crop centre and size, plus a distance map and point samples, in crop-normalised units (0..1)."""
    center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=np.float32)
    size = float(max(box[2] - box[0], box[3] - box[1]) * CROP_PAD)
    scale = CROP_RES / size
    matrix = np.array([[scale, 0, CROP_RES / 2 - center[0] * scale], [0, scale, CROP_RES / 2 - center[1] * scale]], np.float32)
    crop = cv2.warpAffine(mask.astype(np.uint8), matrix, (CROP_RES, CROP_RES), flags=cv2.INTER_NEAREST)
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
    """Masks and PRIMA's starting estimates for each giraffe instance, all as tensors."""

    r0: torch.Tensor  # (N, 3, 3)
    p0: torch.Tensor  # (N, 34, 3, 3)
    t0: torch.Tensor  # (N, 3)
    center: torch.Tensor  # (N, 2)
    size: torch.Tensor  # (N,)
    distance: torch.Tensor  # (N, 1, R, R)
    samples: torch.Tensor  # (N, K, 2)
    track: np.ndarray
    frame: np.ndarray

    def __len__(self) -> int:
        return len(self.r0)


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


def project_to_crop(vertices, cam_t, focal, image_size, center, size) -> torch.Tensor:
    """Vertices (N, V, 3) to crop-normalised coordinates, with 0..1 inside the crop."""
    camera = vertices + cam_t[:, None]
    principal = torch.tensor([image_size[0] / 2, image_size[1] / 2], device=vertices.device, dtype=vertices.dtype)
    pixels = focal * camera[..., :2] / camera[..., 2:].clamp(min=1e-3) + principal
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


def pose_prior(poses: Poses, idx) -> torch.Tensor:
    """Stay near PRIMA's pose, but let the neck and head move: its neck rotations suit a short neck."""
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


def optimize(body, poses: Poses, focal, image_size, iters, batch, lr_pose, lr_shape=0.0, lr_offsets=0.0,
             w_pose=1e-2, w_temporal=0.0, w_laplacian=0.0, w_lateral=0.0, w_scale_prior=1e-3, seed=0, log=print) -> None:
    """Parameters with a zero learning rate stay fixed. Betas stay at their starting value throughout:
    left free they trade off against bone scales and drift far outside SMAL's plausible range."""
    inst = poses.inst
    # Camera corrections are relative to depth, so the same step moves them far more in pixels.
    pose_optimizer = IndexedAdam([(poses.d_orient, lr_pose), (poses.d_pose, lr_pose), (poses.d_cam, lr_pose * 0.1)])
    shape_groups = []
    if lr_shape:
        shape_groups.append({"params": [body.log_scale], "lr": lr_shape})
    if lr_offsets:
        shape_groups.append({"params": [body.offsets], "lr": lr_offsets})
    shape_optimizer = torch.optim.Adam(shape_groups) if shape_groups else None
    generator = torch.Generator(device=inst.r0.device).manual_seed(seed)
    scale_start = body.log_scale.detach().clone()

    for step in range(iters):
        idx = torch.randperm(len(inst), generator=generator, device=inst.r0.device)[:batch]
        orient, pose = poses.matrices(idx)
        vertices = body(orient, pose)
        uv = project_to_crop(vertices, poses.translation(idx), focal, image_size, inst.center[idx], inst.size[idx])
        outside, coverage = mask_loss(uv, inst.distance[idx], inst.samples[idx])

        loss = outside + coverage + w_pose * pose_prior(poses, idx)
        if w_temporal:
            loss = loss + w_temporal * temporal_loss(poses)
        if lr_shape:
            loss = loss + w_scale_prior * (body.log_scale - scale_start).pow(2).mean()
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
        if step % max(iters // 5, 1) == 0 or step == iters - 1:
            log(f"  step {step:5d} outside {outside.item():.5f} coverage {coverage.item():.5f} {body.proportions()}")


@torch.no_grad()
def posed_vertices(body, poses: Poses, chunk: int = 64) -> np.ndarray:
    parts = []
    for start in range(0, len(poses.inst), chunk):
        idx = torch.arange(start, min(start + chunk, len(poses.inst)), device=poses.inst.r0.device)
        parts.append(body(*poses.matrices(idx)).cpu().numpy())
    return np.concatenate(parts)
