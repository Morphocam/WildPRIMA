"""LiDAR gives what the images cannot: true scale, and a check on the mesh that the fit never saw.

The fit uses only masks. LiDAR is compared with the finished mesh, pixel by pixel, where both exist:
  * metric scale: LiDAR depth over mesh depth, i.e. metres per model unit (should be constant per animal);
  * relief error: after removing that scale, how far the mesh's surface shape departs from the LiDAR's.
"""
from __future__ import annotations

import cv2
import numpy as np

MIN_POINTS = 8


def lidar_points(depth_m: np.ndarray, mask: np.ndarray, erode_frac: float = 0.03) -> np.ndarray:
    """Boolean map of LiDAR points well inside a mask. Eroding avoids returns from just past the edge."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros_like(mask, dtype=bool)
    radius = int(round(min(ys.max() - ys.min(), xs.max() - xs.min()) * erode_frac))
    inner = mask
    if radius >= 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        inner = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    return inner & (depth_m > 0)


def instance_depth(depth_m: np.ndarray, mask: np.ndarray) -> tuple[float, int]:
    """Median LiDAR depth inside a mask, and the number of points; NaN when there are too few."""
    values = depth_m[lidar_points(depth_m, mask)]
    return (float(np.median(values)) if len(values) >= MIN_POINTS else float("nan")), len(values)


def mask_height_m(mask: np.ndarray, depth_m: float, fy: float) -> float:
    """Apparent height of a mask in metres at a depth. Uses no 3D model, so it cross-checks the mesh."""
    ys = np.nonzero(mask.any(axis=1))[0]
    return float((ys.max() - ys.min() + 1) * depth_m / fy) if len(ys) else float("nan")


# LiDAR used by the fit is split from LiDAR kept back to check it in a checkerboard of square blocks, so that a
# held-out point is rarely right beside a fitted one. Points on one animal are still not independent of each other.
SPLIT_BLOCK = 16  # pixels
# LiDAR returns this far from the animal's median depth come through gaps (between legs, past the outline).
MAX_RELIEF_M = 2.0


def split_pattern(height: int, width: int) -> np.ndarray:
    """True on the blocks whose LiDAR the fit may use, False on the blocks kept back to check it."""
    ys, xs = np.mgrid[0:height, 0:width]
    return ((ys // SPLIT_BLOCK + xs // SPLIT_BLOCK) % 2) == 0


def fitting_points(depth_m: np.ndarray, mask: np.ndarray, max_points: int, rng: np.random.Generator):
    """Pixel coordinates (x, y) and depths of the LiDAR points on an animal that the fit may use."""
    picks = lidar_points(depth_m, mask) & split_pattern(*depth_m.shape)
    ys, xs = np.nonzero(picks)
    depth = depth_m[ys, xs]
    if len(depth):
        near = np.abs(depth - np.median(depth)) < MAX_RELIEF_M
        ys, xs, depth = ys[near], xs[near], depth[near]
    if len(depth) > max_points:
        keep = rng.choice(len(depth), max_points, replace=False)
        ys, xs, depth = ys[keep], xs[keep], depth[keep]
    return np.stack([xs, ys], axis=1).astype(np.float32), depth.astype(np.float32)


def compare_mesh(depth_m: np.ndarray, mask: np.ndarray, mesh_depth: np.ndarray, where: np.ndarray | None = None) -> dict:
    """Mesh depth map (model units, 0 where absent) against LiDAR, at LiDAR points on both animal and mesh.

    `where` limits the points (to the held-out half, say). `flat_relief_m` is what a flat surface would score.
    """
    result = {"metres_per_unit": float("nan"), "relief_error_m": float("nan"), "flat_relief_m": float("nan"), "lidar_on_mesh": 0}
    picks = lidar_points(depth_m, mask) & (mesh_depth > 0)
    if where is not None:
        picks &= where
    count = int(picks.sum())
    result["lidar_on_mesh"] = count
    if count < MIN_POINTS:
        return result
    lidar, mesh = depth_m[picks], mesh_depth[picks]
    scale = float(np.median(lidar) / np.median(mesh))
    # Depth in front of or behind the animal's middle: the shape of the surface, not its distance.
    residual = (lidar - np.median(lidar)) - scale * (mesh - np.median(mesh))
    result["metres_per_unit"] = scale
    result["relief_error_m"] = float(np.median(np.abs(residual)))
    result["flat_relief_m"] = float(np.median(np.abs(lidar - np.median(lidar))))
    return result
