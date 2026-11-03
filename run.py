#!/usr/bin/env python
"""Recover a 3D mesh per giraffe in a video with PRIMA, and measure how well the mesh size fits.

    python run.py 1.mp4                     # every giraffe found
    python run.py 1.mp4 --max-animals 1     # only the largest giraffe in each frame

Writes to outputs/<video>/: overlay.mp4, dashboard.mp4, metrics.csv, summary.json, fit_vs_size.png,
predictions.npz and meshes/*.obj.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from giraffe3d.detection import GiraffeDetector
from giraffe3d.metrics import fit_metrics, plot_summary, scene_flags, summarize
from giraffe3d.prima_model import PrimaModel
from giraffe3d.render import FrameRenderer, ProfileRenderer, export_mesh, track_color
from giraffe3d.tracking import Tracker

ROOT = Path(__file__).resolve().parent
PANEL = 308
SLOTS = 6
# Measured from the SMAL template's axes (x: tail to nose, y: lateral, z: up): profile from the side.
PROFILE_VIEW = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
PROFILE_HALF_EXTENT = 1.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="Output folder (default outputs/<video stem>)")
    parser.add_argument("--prima-dir", type=Path, default=ROOT / "third_party/PRIMA")
    parser.add_argument("--det-thresh", type=float, default=0.7, help="Giraffe detection score threshold; lower admits partial-body boxes")
    parser.add_argument("--max-animals", type=int, default=0, help="Keep only the N largest giraffes per frame (0 = all)")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many processed frames (0 = all)")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--mesh-stride", type=int, default=10, help="Save OBJ meshes every N source frames (0 = none)")
    return parser.parse_args()


def label(image: np.ndarray, text: str, origin: tuple[int, int], scale: float = 0.5) -> None:
    for color, thickness in (((0, 0, 0), 3), ((255, 255, 255), 1)):
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def square_crop(image: np.ndarray, box: np.ndarray, size: int, pad: float = 1.25) -> np.ndarray:
    """Square region around a box, resized to `size`; areas beyond the frame are black."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    half = max(box[2] - box[0], box[3] - box[1]) * pad / 2
    scale = size / (2 * half)
    matrix = np.array([[scale, 0, size / 2 - cx * scale], [0, scale, size / 2 - cy * scale]], dtype=np.float32)
    return cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)


class SlotMap:
    """Gives each visible track a fixed dashboard column while it remains visible."""

    def __init__(self, count: int):
        self.count = count
        self.assigned: dict[int, int] = {}

    def update(self, track_ids) -> dict[int, int]:
        visible = set(int(t) for t in track_ids)
        self.assigned = {t: s for t, s in self.assigned.items() if t in visible}
        free = [s for s in range(self.count) if s not in self.assigned.values()]
        for track in sorted(visible - set(self.assigned)):
            if free:
                self.assigned[track] = free.pop(0)
        return self.assigned


def build_dashboard(frame_rgb, overlay_rgb, slots: dict[int, tuple[np.ndarray, np.ndarray]],
                    titles: tuple[str, str] = ("Input", "PRIMA meshes")) -> np.ndarray:
    """Top: input | overlay. Below, one column per giraffe: mesh on crop, then fixed-scale profile."""
    width = PANEL * SLOTS
    half = width // 2
    top_height = int(round(half * frame_rgb.shape[0] / frame_rgb.shape[1])) // 2 * 2
    canvas = np.full((top_height + 2 * PANEL, width, 3), 24, dtype=np.uint8)
    canvas[:top_height, :half] = cv2.resize(frame_rgb, (half, top_height), interpolation=cv2.INTER_AREA)
    canvas[:top_height, half:] = cv2.resize(overlay_rgb, (half, top_height), interpolation=cv2.INTER_AREA)
    label(canvas, titles[0], (12, 26), 0.7)
    label(canvas, titles[1], (half + 12, 26), 0.7)
    for slot, (crop, profile) in slots.items():
        canvas[top_height : top_height + PANEL, slot * PANEL : (slot + 1) * PANEL] = crop
        canvas[top_height + PANEL : top_height + 2 * PANEL, slot * PANEL : (slot + 1) * PANEL] = profile
    return canvas


def transcode_h264(source: Path, target: Path, fps: float) -> None:
    """OpenCV only writes mp4v; re-encode so that the video plays in browsers and players."""
    import imageio_ffmpeg

    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-r", f"{fps}", "-i", str(source),
         "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(target)],
        check=True,
    )
    source.unlink()


def main() -> None:
    args = parse_args()
    out_dir = args.out or ROOT / "outputs" / args.video.stem
    mesh_dir = out_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = (cap.get(cv2.CAP_PROP_FPS) or 30.0) / max(1, args.frame_stride)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detector = GiraffeDetector(args.det_thresh)
    model = PrimaModel(args.prima_dir)
    frame_renderer = FrameRenderer(width, height, model.focal_for(width, height), model.faces)
    profile_renderer = ProfileRenderer(PANEL, model.faces, PROFILE_HALF_EXTENT, PROFILE_VIEW)
    tracker = Tracker()
    slot_map = SlotMap(SLOTS)

    raw_overlay, raw_dashboard = out_dir / "_overlay.mp4", out_dir / "_dashboard.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    overlay_writer = cv2.VideoWriter(str(raw_overlay), fourcc, fps, (width, height))
    dashboard_writer = None

    rows: list[dict] = []
    saved: dict[str, list] = {k: [] for k in ("frame", "track_id", "vertices", "cam_t", "betas", "global_orient", "pose")}
    stock_filter_hits = 0
    frames_with_stock_hit = 0
    processed = 0

    for frame_idx in tqdm(range(total), desc="frames"):
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_idx % max(1, args.frame_stride):
            continue
        if args.max_frames and processed >= args.max_frames:
            break
        processed += 1
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        dets, stock_hits = detector(frame_bgr)
        stock_filter_hits += stock_hits
        frames_with_stock_hit += int(stock_hits > 0)
        if args.max_animals and len(dets) > args.max_animals:
            areas = (dets.boxes[:, 2] - dets.boxes[:, 0]) * (dets.boxes[:, 3] - dets.boxes[:, 1])
            dets = dets.select(np.argsort(-areas)[: args.max_animals])
        track_ids = tracker.update(dets.boxes)
        slots = slot_map.update(track_ids)

        colors = [track_color(int(t)) for t in track_ids]
        overlay = frame_rgb.copy()
        panels: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if len(dets):
            pred = model(frame_bgr, dets.boxes)
            overlay = frame_renderer.overlay(frame_rgb, pred.vertices, pred.cam_t, colors)
            truncated, overlapping = scene_flags(dets.boxes, width, height)

            for i, track_id in enumerate(int(t) for t in track_ids):
                box = dets.boxes[i]
                mesh_mask = frame_renderer.silhouette(pred.vertices[i], pred.cam_t[i])
                fit = fit_metrics(mesh_mask, dets.masks[i])
                rows.append(
                    {
                        "frame": frame_idx, "track_id": track_id, "score": float(dets.scores[i]),
                        "x1": float(box[0]), "y1": float(box[1]), "x2": float(box[2]), "y2": float(box[3]),
                        "box_w": float(box[2] - box[0]), "box_h": float(box[3] - box[1]),
                        "cam_x": float(pred.cam_t[i, 0]), "cam_y": float(pred.cam_t[i, 1]), "cam_z": float(pred.cam_t[i, 2]),
                        "truncated": bool(truncated[i]), "overlapping": bool(overlapping[i]),
                        **fit, **pred.shape_stats[i],
                    }
                )
                for key, value in (("frame", frame_idx), ("track_id", track_id), ("vertices", pred.vertices[i]),
                                   ("cam_t", pred.cam_t[i]), ("betas", pred.betas[i]),
                                   ("global_orient", pred.global_orient[i]), ("pose", pred.pose[i])):
                    saved[key].append(value)
                if args.mesh_stride and frame_idx % args.mesh_stride == 0:
                    export_mesh(mesh_dir / f"frame{frame_idx:05d}_giraffe{track_id:02d}.obj",
                                pred.vertices[i], pred.cam_t[i], model.faces, colors[i])

            for i, track_id in enumerate(int(t) for t in track_ids):
                if track_id in slots:
                    crop = square_crop(overlay, dets.boxes[i], PANEL)
                    row = rows[-len(dets) + i]
                    label(crop, f"#{track_id} IoU {row['iou']:.2f}", (8, 20))
                    label(crop, f"h {row['box_h']:.0f}px  mesh/mask {row['height_ratio']:.2f}", (8, 40), 0.45)
                    profile = profile_renderer(pred.vertices[i], pred.global_orient[i], colors[i]).copy()
                    label(profile, f"#{track_id} shape, fixed scale", (8, 20))
                    panels[slots[track_id]] = (crop, profile)

            for i, track_id in enumerate(int(t) for t in track_ids):
                x1, y1 = int(dets.boxes[i][0]), int(dets.boxes[i][1])
                label(overlay, f"#{track_id}", (x1, max(y1 - 6, 16)), 0.7)

        label(overlay, f"frame {frame_idx}  giraffes {len(dets)}", (12, height - 14), 0.7)
        overlay_writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        dashboard = cv2.cvtColor(build_dashboard(frame_rgb, overlay, panels), cv2.COLOR_RGB2BGR)
        if dashboard_writer is None:
            dashboard_writer = cv2.VideoWriter(str(raw_dashboard), fourcc, fps, (dashboard.shape[1], dashboard.shape[0]))
        dashboard_writer.write(dashboard)

    cap.release()
    overlay_writer.release()
    if dashboard_writer is not None:
        dashboard_writer.release()

    transcode_h264(raw_overlay, out_dir / "overlay.mp4", fps)
    transcode_h264(raw_dashboard, out_dir / "dashboard.mp4", fps)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metrics.csv", index=False)
    summary = {
        "stock_prima_filter": {
            "boxes_kept": stock_filter_hits,
            "frames_with_any_box": frames_with_stock_hit,
            "note": "What PRIMA's unmodified demo class filter (no giraffe class) would have kept at score > 0.7.",
        }
    }
    if rows:
        summary.update(summarize(df, processed))
        # Reference: proportions of the mean SMAL animal, and what PRIMA predicted on average.
        template = model.shape_stats(np.zeros((1, len(saved["betas"][0]))))[0]
        summary["shape_vs_mean_smal"] = {
            key: {"mean_smal": template[key], "predicted_median": float(df[key].median())}
            for key in ("neck_to_body", "leg_to_body")
        }
        plot_summary(df, str(out_dir / "fit_vs_size.png"))
        np.savez_compressed(out_dir / "predictions.npz", faces=model.faces,
                            **{k: np.stack([np.asarray(v, dtype=np.float32 if k not in ("frame", "track_id") else np.int32) for v in values]) for k, values in saved.items()})
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Results in {out_dir}")


if __name__ == "__main__":
    main()
