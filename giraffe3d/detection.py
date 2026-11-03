"""Giraffe detection with instance masks.

PRIMA's own demos cannot be used as-is here: ``prima.utils.detection.ANIMAL_COCO_IDS`` omits
COCO class 23 (giraffe), so they silently discard every correctly classified giraffe. They also
use Faster R-CNN; Mask R-CNN (same backbone) additionally gives us masks to score mesh fit against.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GIRAFFE_COCO_ID = 23
STOCK_PRIMA_ANIMAL_IDS = (15, 16, 17, 18, 19, 21, 22)

_CONFIG = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"


@dataclass
class Detections:
    boxes: np.ndarray  # (N, 4) xyxy
    scores: np.ndarray  # (N,)
    masks: np.ndarray  # (N, H, W) bool

    def __len__(self) -> int:
        return len(self.boxes)

    def select(self, idx) -> "Detections":
        return Detections(self.boxes[idx], self.scores[idx], self.masks[idx])


def dedupe_by_mask(
    scores: np.ndarray,
    masks: np.ndarray,
    iou_threshold: float = 0.5,
    containment_threshold: float = 0.85,
) -> np.ndarray:
    """Indices to keep, highest score first, removing duplicate/partial instance detections.

    Compares masks rather than boxes because a distant giraffe can sit entirely inside a near
    giraffe's box, which box-containment suppression (as in PRIMA's demo) would wrongly remove.
    """
    flat = masks.reshape(len(masks), -1)
    areas = flat.sum(axis=1)
    keep: list[int] = []
    for i in np.argsort(-scores):
        if areas[i] == 0:
            continue
        duplicate = False
        for k in keep:
            inter = np.logical_and(flat[i], flat[k]).sum()
            union = areas[i] + areas[k] - inter
            if inter / union > iou_threshold or inter / min(areas[i], areas[k]) > containment_threshold:
                duplicate = True
                break
        if not duplicate:
            keep.append(int(i))
    return np.array(keep, dtype=np.int64)


class GiraffeDetector:
    def __init__(self, score_threshold: float = 0.5, device: str = "cuda"):
        import detectron2.config
        import detectron2.engine
        from detectron2 import model_zoo

        cfg = detectron2.config.get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(_CONFIG))
        # Run the network low so that stock-class counts are comparable; filter per class below.
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = min(score_threshold, 0.5)
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(_CONFIG)
        cfg.MODEL.DEVICE = device
        self.predictor = detectron2.engine.DefaultPredictor(cfg)
        self.score_threshold = score_threshold

    def __call__(self, frame_bgr: np.ndarray) -> tuple[Detections, int]:
        """Return giraffe detections and how many boxes PRIMA's stock filter would have kept."""
        inst = self.predictor(frame_bgr)["instances"].to("cpu")
        classes = inst.pred_classes.numpy()
        scores = inst.scores.numpy()
        stock_hits = int((np.isin(classes, STOCK_PRIMA_ANIMAL_IDS) & (scores > 0.7)).sum())

        idx = np.where((classes == GIRAFFE_COCO_ID) & (scores >= self.score_threshold))[0]
        dets = Detections(
            inst.pred_boxes.tensor.numpy()[idx].astype(np.float32),
            scores[idx],
            inst.pred_masks.numpy()[idx],
        )
        if len(dets) > 1:
            dets = dets.select(dedupe_by_mask(dets.scores, dets.masks))
        return dets, stock_hits
