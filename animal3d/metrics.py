"""Measures how well reprojected meshes match the animals' measured image size.

"Size" is measured against the detector's instance mask, not by eye:
  * silhouette IoU between the mesh's projection and the mask,
  * mesh/mask height, width and area ratios (1.0 = correct image size; area above 1 = too bulky),
  * top recall: fraction of the mask's top third covered by the mesh. On a standing animal that is the
    head, neck and back, where a wrong body plan shows first (on a giraffe it isolates the long neck).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SIZE_BUCKETS = ((0, 150, "small (<150 px)"), (150, 250, "medium (150-250 px)"), (250, 1e9, "large (>=250 px)"))


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def fit_metrics(mesh_mask: np.ndarray, det_mask: np.ndarray) -> dict:
    """Compare a mesh silhouette against a detector mask of the same image size."""
    result = {"iou": 0.0, "height_ratio": np.nan, "width_ratio": np.nan, "area_ratio": np.nan, "top_recall": np.nan}
    det_box = mask_bbox(det_mask)
    if det_box is None:
        return result
    result["area_ratio"] = float(mesh_mask.sum() / det_mask.sum())
    inter = np.logical_and(mesh_mask, det_mask).sum()
    union = np.logical_or(mesh_mask, det_mask).sum()
    result["iou"] = float(inter / union) if union else 0.0

    x1, y1, x2, y2 = det_box
    top = det_mask.copy()
    top[y1 + (y2 - y1) // 3 :] = False
    result["top_recall"] = float(np.logical_and(mesh_mask, top).sum() / max(top.sum(), 1))

    mesh_box = mask_bbox(mesh_mask)
    if mesh_box is not None:
        result["height_ratio"] = (mesh_box[3] - mesh_box[1]) / max(y2 - y1, 1)
        result["width_ratio"] = (mesh_box[2] - mesh_box[0]) / max(x2 - x1, 1)
    return result


def scene_flags(boxes: np.ndarray, width: int, height: int, margin: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Per box: touches the frame edge; overlaps another animal's box. Both invalidate the metrics."""
    truncated = (
        (boxes[:, 0] <= margin) | (boxes[:, 1] <= margin) | (boxes[:, 2] >= width - margin) | (boxes[:, 3] >= height - margin)
    )
    overlapping = np.zeros(len(boxes), dtype=bool)
    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if i != j:
                w = min(boxes[i, 2], boxes[j, 2]) - max(boxes[i, 0], boxes[j, 0])
                h = min(boxes[i, 3], boxes[j, 3]) - max(boxes[i, 1], boxes[j, 1])
                overlapping[i] |= w > 0 and h > 0
    return truncated, overlapping


def _aggregate(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0}
    return {
        "n": int(len(df)),
        "iou_median": float(df["iou"].median()),
        "iou_p10": float(df["iou"].quantile(0.1)),
        "height_ratio_median": float(df["height_ratio"].median()),
        "width_ratio_median": float(df["width_ratio"].median()),
        "area_ratio_median": float(df["area_ratio"].median()),
        "top_recall_median": float(df["top_recall"].median()),
        "frac_height_within_15pct": float(((df["height_ratio"] - 1).abs() <= 0.15).mean()),
    }


def summarize(df: pd.DataFrame, frames_total: int, min_track_len: int = 10) -> dict:
    """Aggregate fit by size bucket; per track, check that apparent-size change is explained by depth."""
    clean = df[~df["truncated"] & ~df["overlapping"]]
    summary: dict = {
        "frames_processed": frames_total,
        "frames_with_animal": int(df["frame"].nunique()),
        "animal_instances": int(len(df)),
        "max_animals_in_one_frame": int(df.groupby("frame").size().max()) if len(df) else 0,
        "tracks": int(df["track_id"].nunique()),
        "all": _aggregate(df),
        "clean": _aggregate(clean),
        "clean_by_size": {
            name: _aggregate(clean[(clean["box_h"] >= lo) & (clean["box_h"] < hi)]) for lo, hi, name in SIZE_BUCKETS
        },
        "per_track": [],
    }
    for track_id, group in df.groupby("track_id"):
        if len(group) < min_track_len:
            continue
        inverse_depth = 1.0 / group["cam_z"]
        vary = group["box_h"].std() > 0 and inverse_depth.std() > 0
        summary["per_track"].append(
            {
                "track_id": int(track_id),
                "frames": int(len(group)),
                "box_h_min": float(group["box_h"].min()),
                "box_h_max": float(group["box_h"].max()),
                "iou_median": float(group["iou"].median()),
                # Same animal should keep the same shape: low CV = shape does not chase apparent size.
                "neck_to_body_mean": float(group["neck_to_body"].mean()),
                "neck_to_body_cv": float(group["neck_to_body"].std() / group["neck_to_body"].mean()),
                "leg_to_body_cv": float(group["leg_to_body"].std() / group["leg_to_body"].mean()),
                # Near 1: when the animal looks bigger the model moves it closer, as it should.
                "box_h_vs_inverse_depth_corr": float(np.corrcoef(group["box_h"], inverse_depth)[0, 1]) if vary else None,
            }
        )
    return summary


def plot_summary(df: pd.DataFrame, path: str) -> None:
    """Single-measure panels against the same x axis: animal height in the image."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, muted, grid, series = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6"
    clean = df[~df["truncated"] & ~df["overlapping"]]
    panels = [
        ("iou", "Silhouette IoU with detector mask", (0, 1), None),
        ("height_ratio", "Mesh height / mask height", (0.4, 2.0), 1.0),
        ("area_ratio", "Mesh area / mask area", (0.4, 2.0), 1.0),
        ("top_recall", "Top recall (mask top third)", (0, 1), None),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), facecolor=surface)
    for ax, (column, title, ylim, reference) in zip(axes, panels):
        ax.set_facecolor(surface)
        ax.scatter(clean["box_h"], clean[column], s=14, color=series, alpha=0.55, edgecolors=surface, linewidths=0.4)
        if reference is not None:
            ax.axhline(reference, color=muted, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_ylim(*ylim)
        ax.set_title(title, loc="left", fontsize=11, color=ink)
        ax.set_xlabel("Animal height in image (px)", fontsize=9, color=muted)
        ax.tick_params(colors=muted, labelsize=8, length=0)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(grid)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)
