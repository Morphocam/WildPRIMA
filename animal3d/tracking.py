"""Small constant-velocity tracker giving each animal a stable ID across frames.

Animals can move far between frames at this video's ~10 fps, so plain IoU is
not enough: matching also accepts boxes whose centres are close relative to the box size.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a[:, None, :]
    b = b[None, :, :]
    w = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    h = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = w * h
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-6)


@dataclass
class _Track:
    id: int
    box: np.ndarray
    velocity: np.ndarray
    misses: int = 0

    def predicted(self) -> np.ndarray:
        return self.box + self.velocity * (self.misses + 1)


class Tracker:
    def __init__(self, max_misses: int = 15, max_center_dist: float = 0.6, velocity_smoothing: float = 0.5):
        self.max_misses = max_misses
        self.max_center_dist = max_center_dist  # in units of the track box's larger side
        self.velocity_smoothing = velocity_smoothing
        self.tracks: list[_Track] = []
        self._next_id = 1

    def update(self, boxes: np.ndarray) -> np.ndarray:
        """Return a track ID for each row of ``boxes``."""
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        ids = np.full(len(boxes), -1, dtype=np.int64)
        unmatched_tracks = set(range(len(self.tracks)))

        if len(boxes) and self.tracks:
            pred = np.stack([t.predicted() for t in self.tracks])
            iou = iou_matrix(pred, boxes)
            pred_c = (pred[:, :2] + pred[:, 2:]) / 2
            box_c = (boxes[:, :2] + boxes[:, 2:]) / 2
            sizes = np.maximum(pred[:, 2] - pred[:, 0], pred[:, 3] - pred[:, 1])[:, None]
            dist = np.linalg.norm(pred_c[:, None] - box_c[None], axis=2) / np.maximum(sizes, 1.0)
            feasible = (iou > 0.05) | (dist < self.max_center_dist)
            cost = (1 - iou) + dist
            cost[~feasible] = 1e6
            for ti, di in zip(*linear_sum_assignment(cost)):
                if not feasible[ti, di]:
                    continue
                track = self.tracks[ti]
                step = (boxes[di] - track.box) / (track.misses + 1)
                s = self.velocity_smoothing
                track.velocity = s * track.velocity + (1 - s) * step
                track.box = boxes[di].copy()
                track.misses = 0
                ids[di] = track.id
                unmatched_tracks.discard(ti)

        for ti in unmatched_tracks:
            self.tracks[ti].misses += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

        for di in np.where(ids < 0)[0]:
            self.tracks.append(_Track(self._next_id, boxes[di].copy(), np.zeros(4, dtype=np.float32)))
            ids[di] = self._next_id
            self._next_id += 1
        return ids
