#!/usr/bin/env python
"""Build a 3D model of the animals in an image, a folder of images, a video or an RGB + LiDAR snippet.

    python run.py photo.jpg
    python run.py photos/                      # one species per run: all animals share one body shape
    python run.py 1.mp4 --species giraffe
    python run.py input_30s/giraffe_4          # rgb/ + depthm/ + metadata.txt: adds sizes in metres
    python run.py 1.mp4 --baseline-only        # PRIMA alone, without measuring the body

The body is measured from the masks alone. The species (from --species, metadata.txt or BioCLIP) only chooses which
horns to try: paired, on the nose, or none. Writes to outputs/<input name>/.
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import warnings
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from animal3d.body import BODY_BOUNDS, AnimalBody, make_init
from animal3d.detection import COCO_ANIMALS, AnimalDetector
from animal3d.fit import Instances, Poses, choose_shape_instances, fit_body, optimize, posed_vertices, prepare_mask, search_body
from animal3d.lidar import compare_mesh, fitting_points, instance_depth, mask_height_m, split_pattern
from animal3d.metrics import fit_metrics, plot_summary, scene_flags
from animal3d.render import FrameRenderer, ProfileRenderer, export_mesh, track_color, turntable_frames
from animal3d.species import BY_KEY, decide
from animal3d.sources import Camera, FrameSource, move_to_camera, open_source, scaled_camera
from animal3d.tracking import Tracker

ROOT = Path(__file__).resolve().parent
PANEL = 308
SLOTS = 6
# Measured from the SMAL template's axes (x: tail to nose, y: lateral, z: up): profile from the side.
PROFILE_VIEW = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
# From PRIMA's checkpoint config (EXTRA.FOCAL_LENGTH / MODEL.IMAGE_SIZE), for use without loading it.
PRIMA_FOCAL_RATIO = 1000.0 / 256.0
FIT_COLUMNS = ("iou", "height_ratio", "width_ratio", "area_ratio", "top_recall")
SIDE_ON_LIMIT = 0.6  # |cos| between the body's long axis and the view direction
MIN_LIDAR_INSTANCES = 10
EAR_THICKNESS = 1.6  # a fixed short ear stays about as broad as SMAL's, so it ends up rounder
MIN_SHAPE_BOX_H = 40  # pixels: smaller animals are not used to measure the body


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Image, video, folder of images, or snippet folder (rgb/ + depthm/)")
    parser.add_argument("--species", default=None, help="Label for the outputs (default: metadata.txt, else 'animal')")
    parser.add_argument("--horns", choices=("auto", "masks", "none", "paired", "nasal_two", "nasal_one"), default="auto",
                        help="auto: from the species (label, else BioCLIP), else as 'masks'; masks: paired horns only if the masks "
                             "clearly support them; or force a layout")
    parser.add_argument("--humps", choices=("auto", "0", "1", "2"), default="auto", help="auto: from the species (camels)")
    parser.add_argument("--ears", default="auto",
                        help="ear length against SMAL's: auto (from the species, else fitted), masks (fitted), smal (SMAL's ears, unchanged), or a number")
    parser.add_argument("--depth-weight", type=float, default=0.3,
                        help="Weight of LiDAR depth relief when posing each animal (half its points; the other half checks it). 0: masks only")
    parser.add_argument("--out", type=Path, default=None, help="Output folder (default outputs/<input name>)")
    parser.add_argument("--calibration", default="auto", help="auto, none, longleat_v2, bubye, or a JSON file; see animal3d/sources.py")
    parser.add_argument("--prima-dir", type=Path, default=ROOT / "third_party/PRIMA")
    parser.add_argument("--det-thresh", type=float, default=0.7, help="Detection score threshold; lower finds more but admits partial-body boxes")
    parser.add_argument("--max-animals", type=int, default=0, help="Keep only the N largest animals per frame (0 = all)")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many processed frames (0 = all)")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--mesh-stride", type=int, default=10, help="Save OBJ meshes every N source frames (0 = none)")
    parser.add_argument("--baseline-only", action="store_true", help="Stop after PRIMA; do not measure the body")
    parser.add_argument("--reuse-baseline", action="store_true", help="Reuse the detections and PRIMA results already in the output folder")
    parser.add_argument("--no-videos", action="store_true", help="Skip the overlay and comparison output")
    parser.add_argument("--ablation", choices=("none", "smal", "betas", "nosearch"), default="none",
                        help="Ablations of the body model (paper): smal = PRIMA's SMAL shape, pose refined only; betas = SMAL shape "
                             "coefficients fitted instead of bone scales; nosearch = bone scales from PRIMA's plan, no body-plan search. "
                             "Combine with --horns none --humps 0 --ears smal and --offset-iters 0 for the other rows")
    parser.add_argument("--bone-iters", type=int, default=2000)
    parser.add_argument("--offset-iters", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=48)
    return parser.parse_args()


# ---------------------------------------------------------------------------------------- drawing


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


def build_dashboard(left_rgb, right_rgb, slots: dict[int, tuple[np.ndarray, np.ndarray]], titles: tuple[str, str]) -> np.ndarray:
    """Top: two views of the frame. Below, one column per animal: mesh on crop, then fixed-scale profile."""
    width = PANEL * SLOTS
    half = width // 2
    top_height = int(round(half * left_rgb.shape[0] / left_rgb.shape[1])) // 2 * 2
    canvas = np.full((top_height + 2 * PANEL, width, 3), 24, dtype=np.uint8)
    canvas[:top_height, :half] = cv2.resize(left_rgb, (half, top_height), interpolation=cv2.INTER_AREA)
    canvas[:top_height, half:] = cv2.resize(right_rgb, (half, top_height), interpolation=cv2.INTER_AREA)
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


class Renderers:
    """One renderer per frame size and camera; they last as long as the process (see FrameRenderer)."""

    def __init__(self, faces: np.ndarray):
        self.faces = faces
        self.cache: dict[tuple, FrameRenderer] = {}

    def get(self, width: int, height: int, camera: Camera) -> FrameRenderer:
        key = (width, height, camera)
        if key not in self.cache:
            self.cache[key] = FrameRenderer(width, height, camera, self.faces)
        return self.cache[key]


def prima_camera(width: int, height: int) -> Camera:
    focal = PRIMA_FOCAL_RATIO * max(width, height)
    return Camera(focal, focal, width / 2, height / 2)


def fit_camera(source: FrameSource, width: int, height: int) -> Camera:
    """The real camera if known, else PRIMA's assumed one."""
    return scaled_camera(source.calibration, width, height) if source.calibration else prima_camera(width, height)


# ------------------------------------------------------------------------------------ PRIMA stage


def full_mask(entry, height: int, width: int) -> np.ndarray:
    crop, (y0, x0) = entry
    mask = np.zeros((height, width), dtype=bool)
    mask[y0 : y0 + crop.shape[0], x0 : x0 + crop.shape[1]] = crop
    return mask


def crop_mask(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    y0, x0 = int(ys.min()), int(xs.min())
    return mask[y0 : ys.max() + 1, x0 : xs.max() + 1].copy(), (y0, x0)


def run_prima(source: FrameSource, args, out: Path):
    """Detect, track and run PRIMA on every frame. Returns (rows, predictions, masks), and saves them."""
    from animal3d.prima_model import PrimaModel

    detector = AnimalDetector(args.det_thresh)
    model = PrimaModel(args.prima_dir)
    tracker = Tracker()
    next_still_id = 1
    rows: list[dict] = []
    masks: list = []
    saved: dict[str, list] = {k: [] for k in ("vertices", "cam_t", "betas", "global_orient", "pose")}
    stats = {"frames_processed": 0, "stock_prima_boxes": 0, "stock_prima_frames": 0}

    for frame in tqdm(source, desc="detect + PRIMA", total=len(source.images) or None):
        if frame.index % max(1, args.frame_stride):
            continue
        if args.max_frames and stats["frames_processed"] >= args.max_frames:
            break
        stats["frames_processed"] += 1
        height, width = frame.bgr.shape[:2]
        dets, stock_hits = detector(frame.bgr)
        stats["stock_prima_boxes"] += stock_hits
        stats["stock_prima_frames"] += int(stock_hits > 0)
        if args.max_animals and len(dets) > args.max_animals:
            areas = (dets.boxes[:, 2] - dets.boxes[:, 0]) * (dets.boxes[:, 3] - dets.boxes[:, 1])
            dets = dets.select(np.argsort(-areas)[: args.max_animals])
        if source.sequence:
            track_ids = tracker.update(dets.boxes)
        else:  # unrelated photos: no animal continues from one to the next
            track_ids = np.arange(next_still_id, next_still_id + len(dets))
            next_still_id += len(dets)
        if not len(dets):
            continue

        pred = model(frame.bgr, dets.boxes)
        truncated, overlapping = scene_flags(dets.boxes, width, height)
        camera = fit_camera(source, width, height)
        for i, track_id in enumerate(int(t) for t in track_ids):
            box = dets.boxes[i]
            lidar_depth, lidar_count = (float("nan"), 0)
            apparent_height = float("nan")
            if frame.depth_m is not None:
                lidar_depth, lidar_count = instance_depth(frame.depth_m, dets.masks[i])
                apparent_height = mask_height_m(dets.masks[i], lidar_depth, camera.fy)
            rows.append({
                "frame": frame.index, "name": frame.name, "track_id": track_id, "width": width, "height": height,
                "score": float(dets.scores[i]), "coco_class": COCO_ANIMALS[int(dets.classes[i])],
                "x1": float(box[0]), "y1": float(box[1]), "x2": float(box[2]), "y2": float(box[3]),
                "box_h": float(box[3] - box[1]), "truncated": bool(truncated[i]), "overlapping": bool(overlapping[i]),
                "lidar_depth_m": lidar_depth, "lidar_points": lidar_count, "mask_height_m": apparent_height,
                **pred.shape_stats[i],
            })
            masks.append(crop_mask(dets.masks[i]))
            for key, value in (("vertices", pred.vertices), ("cam_t", pred.cam_t), ("betas", pred.betas),
                               ("global_orient", pred.global_orient), ("pose", pred.pose)):
                saved[key].append(value[i])

    df = pd.DataFrame(rows)
    predictions = {k: np.stack(v).astype(np.float32) for k, v in saved.items()} if rows else {}
    if rows:
        predictions["faces"] = model.faces
        np.savez_compressed(out / "baseline.npz", **predictions)
        (out / "masks.pkl").write_bytes(pickle.dumps(masks))
        df.to_csv(out / "detections.csv", index=False)
    (out / "run_stats.json").write_text(json.dumps(stats))
    return df, predictions, masks, stats


def load_prima_results(out: Path):
    for name in ("baseline.npz", "masks.pkl", "detections.csv", "run_stats.json"):
        if not (out / name).exists():
            raise SystemExit(f"--reuse-baseline: {out / name} is missing; run once without it.")
    return (pd.read_csv(out / "detections.csv"), dict(np.load(out / "baseline.npz")),
            pickle.loads((out / "masks.pkl").read_bytes()), json.loads((out / "run_stats.json").read_text()))


# ------------------------------------------------------------------------------------- body stage


def load_smal(prima_dir: Path) -> dict:
    from chumpy.ch import materialize

    with open(prima_dir / "data/smal/my_smpl_00781_4_all.pkl", "rb") as f:
        smal = pickle.load(f, encoding="latin1")
    parents = np.array(smal["kintree_table"][0]).astype(np.int64)
    parents[parents > 1000] = -1  # the root's parent is stored as an unsigned -1
    return {
        "v_template": materialize(smal["v_template"]), "shapedirs": materialize(smal["shapedirs"])[:, :, :41],
        "j_regressor": smal["J_regressor"].toarray(), "parents": parents, "weights": materialize(smal["weights"]),
        "faces": materialize(smal["f"], dtype=np.int32),
    }


def choose_horns(source: FrameSource, df: pd.DataFrame, rows: np.ndarray, mode: str, n_crops: int = 16) -> tuple[str, dict]:
    """Horn mode for the search. For "auto": the species from the label, else BioCLIP on the largest clean views."""
    if mode != "auto":
        return mode, {"mode": mode, "humps": 0, "ears": None}
    label = source.species if source.species and source.species.lower() != "animal" else None
    probabilities = None
    try:
        from animal3d.species import SpeciesClassifier

        picks = df.loc[rows].sort_values("box_h", ascending=False).head(n_crops)
        wanted = {frame: g for frame, g in picks.groupby("frame")}
        crops = []
        for frame in source:
            for _, r in wanted.get(frame.index, pd.DataFrame()).iterrows():
                x1, y1, x2, y2 = (int(r[k]) for k in ("x1", "y1", "x2", "y2"))
                pad = int(0.1 * max(x2 - x1, y2 - y1))
                h, w = frame.bgr.shape[:2]
                crops.append(frame.bgr[max(y1 - pad, 0):min(y2 + pad, h), max(x1 - pad, 0):min(x2 + pad, w), ::-1])
            if len(crops) >= len(picks):
                break
        if crops:
            probabilities = SpeciesClassifier()(crops)
    except Exception as error:  # no weights offline, say: the label alone, or the masks, decide
        print(f"Species classifier unavailable ({type(error).__name__}: {error})")
    decision = decide(label, probabilities)
    info = {"mode": "auto", "label": label, "layout": decision.layout, "reason": decision.reason, "humps": decision.humps,
            "ears": decision.ears}
    if decision.species:
        s = BY_KEY[decision.species]
        info.update(species=s.common, scientific=s.scientific, horns=s.horns, source=s.source)
    if probabilities:
        top = sorted(probabilities.items(), key=lambda kv: -kv[1])[:5]
        info["classifier_top5"] = {BY_KEY[k].common: round(v, 3) for k, v in top}
    return decision.layout or "masks", info


def build_instances(pred, df, masks, source: FrameSource, device) -> Instances:
    rng = np.random.default_rng(0)
    prepared, cameras, translations = [], [], []
    for i in range(len(df)):
        row = df.iloc[i]
        width, height = int(row["width"]), int(row["height"])
        box = row[["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
        prepared.append(prepare_mask(full_mask(masks[i], height, width), box, rng))
        camera = fit_camera(source, width, height)
        cameras.append((camera.fx, camera.fy, camera.cx, camera.cy))
        translations.append(move_to_camera(pred["cam_t"][i], PRIMA_FOCAL_RATIO * max(width, height), width, height, camera)[0])
    as_tensor = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
    return Instances(
        r0=as_tensor(pred["global_orient"]), p0=as_tensor(pred["pose"]), t0=as_tensor(translations), camera=as_tensor(cameras),
        center=as_tensor([p[0] for p in prepared]), size=as_tensor([p[1] for p in prepared]),
        distance=as_tensor([p[2] for p in prepared])[:, None], samples=as_tensor([p[3] for p in prepared]),
        track=df["track_id"].to_numpy(), frame=df["frame"].to_numpy(),
    )


def attach_lidar(inst: Instances, source: FrameSource, df: pd.DataFrame, masks, max_points: int = 256) -> Instances:
    """Adds the LiDAR points the fit may use (one half of a checkerboard split) to every instance that has some."""
    rng = np.random.default_rng(0)
    n = len(df)
    uv = np.zeros((n, max_points, 2), np.float32)
    depth = np.zeros((n, max_points), np.float32)
    ok = np.zeros((n, max_points), bool)
    groups = {frame: group.index.to_numpy() for frame, group in df.groupby("frame")}
    center, size = inst.center.cpu().numpy(), inst.size.cpu().numpy()
    for frame in tqdm(source, desc="lidar for the fit", total=len(source.images) or None):
        if frame.depth_m is None:
            continue
        for row in groups.get(frame.index, ()):
            mask = full_mask(masks[row], int(df.at[row, "height"]), int(df.at[row, "width"]))
            xy, d = fitting_points(frame.depth_m, mask, max_points, rng)
            k = len(d)
            uv[row, :k] = (xy - center[row]) / size[row] + 0.5
            depth[row, :k], ok[row, :k] = d, True
    device = inst.r0.device
    inst.lidar_uv, inst.lidar_d = torch.as_tensor(uv, device=device), torch.as_tensor(depth, device=device)
    inst.lidar_ok = torch.as_tensor(ok, device=device)
    print(f"LiDAR for the fit: {int((ok.sum(1) >= 8).sum())} of {n} instances, median {int(np.median(ok.sum(1)[ok.sum(1) > 0])) if ok.any() else 0} points")
    return inst


def score_meshes(source: FrameSource, df, masks, faces, vertices, cam_t, camera_for) -> pd.DataFrame:
    """Silhouette measures and, where there is LiDAR, the comparison with it, for each instance."""
    renderers = Renderers(faces)
    groups = {frame: group.index.to_numpy() for frame, group in df.groupby("frame")}
    results: dict[int, dict] = {}
    for frame in tqdm(source, desc="scoring", total=len(source.images) or None):
        for row in groups.get(frame.index, ()):
            width, height = int(df.at[row, "width"]), int(df.at[row, "height"])
            renderer = renderers.get(width, height, camera_for(width, height))
            mesh_depth = renderer.depth(vertices[row], cam_t[row])
            mask = full_mask(masks[row], height, width)
            results[row] = fit_metrics(mesh_depth > 0, mask)
            if frame.depth_m is not None:
                results[row].update(compare_mesh(frame.depth_m, mask, mesh_depth))
                # The half of the LiDAR the fit never sees (see animal3d.lidar.split_pattern).
                held = compare_mesh(frame.depth_m, mask, mesh_depth, where=~split_pattern(height, width))
                results[row].update({"relief_error_held_m": held["relief_error_m"], "flat_relief_held_m": held["flat_relief_m"]})
    return pd.DataFrame.from_dict(results, orient="index").sort_index()


def medians(base: pd.DataFrame, fit: pd.DataFrame, where: np.ndarray) -> dict:
    return {"n": int(where.sum()), **{c: {"prima": float(base.loc[where, c].median()), "fitted": float(fit.loc[where, c].median())} for c in FIT_COLUMNS}}


def mean_rotations(matrices: np.ndarray) -> np.ndarray:
    """Chordal mean over the first axis, for rotation matrices of shape (N, ..., 3, 3)."""
    u, _, vt = np.linalg.svd(matrices.mean(axis=0))
    sign = np.sign(np.linalg.det(u @ vt))
    u[..., :, 2] *= sign[..., None]
    return u @ vt


def track_scales(df: pd.DataFrame, clean: np.ndarray) -> pd.DataFrame:
    """Per track with enough LiDAR: metres per model unit, and how steady it is across frames."""
    good = df[clean & df["metres_per_unit"].notna()]
    rows = []
    for track, group in good.groupby("track_id"):
        if len(group) >= MIN_LIDAR_INSTANCES:
            scale = group["metres_per_unit"]
            rows.append({"track_id": int(track), "instances": len(group), "metres_per_unit": float(scale.median()),
                         "scale_spread": float((scale.quantile(0.75) - scale.quantile(0.25)) / scale.median()),
                         "depth_median_m": float(group["lidar_depth_m"].median()),
                         "mask_height_p90_m": float(group["mask_height_m"].quantile(0.9))})
    return pd.DataFrame(rows)


def bake_colors(source, vertices, cam_t, df, fit_df, masks, renderers, camera_for, mirror, min_iou=0.6, cap=300) -> np.ndarray:
    """Per-vertex color (0..1): median over frames where the vertex is visible and inside the mask."""
    good = df.index[fit_df["iou"] >= min_iou].to_numpy()
    if not len(good):
        good = df.index[np.argsort(-fit_df["iou"].to_numpy())[:10]].to_numpy()
    if len(good) > cap:
        good = good[np.linspace(0, len(good) - 1, cap).astype(int)]
    by_frame = {f: g.index.to_numpy() for f, g in df.loc[good].groupby("frame")}
    samples = np.full((len(good), vertices.shape[1], 3), np.nan, dtype=np.float32)
    slot = {row: i for i, row in enumerate(good)}
    for frame in tqdm(source, desc="colors", total=len(source.images) or None):
        if frame.index not in by_frame:
            continue
        rgb = cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        for row in by_frame[frame.index]:
            width, height = int(df.at[row, "width"]), int(df.at[row, "height"])
            camera = camera_for(width, height)
            points = vertices[row] + cam_t[row]
            uv = np.array([camera.fx, camera.fy]) * points[:, :2] / points[:, 2:] + np.array([camera.cx, camera.cy])
            px = np.clip(np.round(uv).astype(int), 0, [width - 1, height - 1])
            depth = renderers.get(width, height, camera).depth(vertices[row], cam_t[row])[px[:, 1], px[:, 0]]
            mask = full_mask(masks[row], height, width)[px[:, 1], px[:, 0]]
            visible = mask & (depth > 0) & (np.abs(depth - points[:, 2]) < 0.05)
            samples[slot[row], visible] = rgb[px[visible, 1], px[visible, 0]]
    with warnings.catch_warnings():  # vertices never seen in any view are NaN, and filled in below
        warnings.simplefilter("ignore", RuntimeWarning)
        colors = np.nanmedian(samples, axis=0)
    colors = np.where(np.isnan(colors), colors[mirror], colors)  # a side never seen borrows the other's
    colors = np.where(np.isnan(colors), np.nanmedian(colors, axis=0), colors)
    return np.clip(colors, 0, 1)


def write_model(out: Path, body, pose, reliable, colors, metres_per_unit: float | None) -> dict:
    """The standalone model in its mean pose, in metres if known, and its measurements."""
    import trimesh

    device = body.log_scale.device
    standing_pose = torch.as_tensor(mean_rotations(pose[reliable if reliable.any() else slice(None)]), device=device, dtype=torch.float32)[None]
    eye = torch.eye(3, device=device)[None]
    with torch.no_grad():
        standing = body(eye, standing_pose)[0].cpu().numpy()
        joints = body.posed_joints(eye, standing_pose)[0].cpu().numpy()
    unit = metres_per_unit or 1.0

    # z-up body frame to y-up, as glTF viewers expect.
    y_up = standing @ PROFILE_VIEW.T * unit
    mesh = trimesh.Trimesh(y_up, body.faces, vertex_colors=np.c_[colors, np.ones(len(colors))], process=False)
    mesh.export(out / "model.glb")
    mesh.export(out / "model.obj")

    raw = out / "_turntable.mp4"
    writer = None
    for frame in turntable_frames(standing, body.faces, colors):
        if writer is None:
            writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 24, (frame.shape[1], frame.shape[0]))
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    transcode_h264(raw, out / "model_turntable.mp4", 24)

    # Sizes describe the body: horns are reported on their own and must not inflate height or length.
    frame_only = standing[: body.n_verts]
    ground = frame_only[:, 2].min()
    extents = frame_only.max(axis=0) - frame_only.min(axis=0)
    sizes = {
        "shoulder_height": float(joints[6, 2] - ground),  # top of the spine; bone, not fur
        "top_height": float(frame_only[:, 2].max() - ground),
        "length": float(extents[0]),
        "width": float(extents[1]),
        **{(name if name.endswith("_height") else f"{name}_length"): value for name, value in body.measurements().items()},
    }
    return {key: value * unit for key, value in sizes.items()}


# ----------------------------------------------------------------------------------------- output


def write_visuals(source: FrameSource, out: Path, df, faces, base_vertices, base_t, fit_vertices=None, fit_t=None,
                  fit_orient=None, fit_df=None, fit_faces=None, profile_extent: float = 1.1) -> None:
    """Overlay and, when there is a fit, the side-by-side comparison. Videos for sequences, JPEGs for photos."""
    base_renderers = Renderers(faces)
    fit_renderers = Renderers(fit_faces if fit_faces is not None else faces)
    profile = ProfileRenderer(PANEL, fit_faces if fit_faces is not None else faces, profile_extent, PROFILE_VIEW)
    slots = SlotMap(SLOTS)
    groups = {frame: group.index.to_numpy() for frame, group in df.groupby("frame")}
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    raw_overlay, raw_compare = out / "_overlay.mp4", out / "_comparison.mp4"
    overlay_writer, compare_writer = None, None
    still_dir = out / "overlays"
    if not source.sequence:
        still_dir.mkdir(exist_ok=True)

    for frame in tqdm(source, desc="visuals", total=len(source.images) or None):
        rows = groups.get(frame.index, np.array([], dtype=int))
        if not source.sequence and not len(rows):
            continue
        height, width = frame.bgr.shape[:2]
        rgb = cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2RGB)
        tracks = df.loc[rows, "track_id"].to_numpy()
        colors = [track_color(int(t)) for t in tracks]
        base = base_renderers.get(width, height, prima_camera(width, height)).overlay(rgb, base_vertices[rows], base_t[rows], colors)
        shown = base
        canvas = None
        if fit_vertices is not None:
            fitted = fit_renderers.get(width, height, fit_camera(source, width, height)).overlay(rgb, fit_vertices[rows], fit_t[rows], colors)
            shown = fitted
            assigned = slots.update(tracks)
            panels = {}
            for row, track, color in zip(rows, tracks, colors):
                if int(track) not in assigned:
                    continue
                box = df.loc[row, ["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
                crop = square_crop(fitted, box, PANEL)
                label(crop, f"#{track} IoU {fit_df.loc[row, 'iou']:.2f}", (8, 20))
                view = profile(fit_vertices[row], fit_orient[row], color).copy()
                label(view, f"#{track} body, fixed scale", (8, 20))
                panels[assigned[int(track)]] = (crop, view)
            canvas = cv2.cvtColor(build_dashboard(base, fitted, panels, ("PRIMA as-is", f"Fitted {source.species} body")), cv2.COLOR_RGB2BGR)

        shown_bgr = cv2.cvtColor(shown, cv2.COLOR_RGB2BGR)
        if source.sequence:
            if overlay_writer is None:
                overlay_writer = cv2.VideoWriter(str(raw_overlay), fourcc, source.fps, (width, height))
            overlay_writer.write(shown_bgr)
            if canvas is not None:
                if compare_writer is None:
                    compare_writer = cv2.VideoWriter(str(raw_compare), fourcc, source.fps, (canvas.shape[1], canvas.shape[0]))
                compare_writer.write(canvas)
        else:
            cv2.imwrite(str(still_dir / f"{frame.name}_overlay.jpg"), shown_bgr)
            if canvas is not None:
                cv2.imwrite(str(still_dir / f"{frame.name}_comparison.jpg"), canvas)

    for writer, raw, name in ((overlay_writer, raw_overlay, "overlay.mp4"), (compare_writer, raw_compare, "comparison.mp4")):
        if writer is not None:
            writer.release()
            transcode_h264(raw, out / name, source.fps)


def main() -> None:
    args = parse_args()
    source = open_source(args.input, args.species, args.calibration)
    out = args.out or ROOT / "outputs" / source.name
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    print(f"Input: {source.describe()}")

    if args.reuse_baseline:
        df, pred, masks, stats = load_prima_results(out)
    else:
        df, pred, masks, stats = run_prima(source, args, out)
    summary: dict = {"input": source.describe(), "run": stats, "instances": int(len(df))}
    stock_note = "boxes PRIMA's unmodified class filter would keep at score > 0.7"
    summary["run"]["stock_prima_note"] = stock_note

    if df.empty:
        summary["result"] = "no animals detected"
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print("No animals detected. Try a lower detection threshold (--det-thresh).")
        return

    summary["coco_class_counts"] = df["coco_class"].value_counts().to_dict()
    summary["frames_with_animal"] = int(df["frame"].nunique())
    summary["tracks"] = int(df["track_id"].nunique())
    summary["max_animals_in_one_frame"] = int(df.groupby("frame").size().max())
    faces = pred["faces"].astype(np.int64)

    print("Scoring PRIMA as-is")
    base_df = score_meshes(source, df, masks, faces, pred["vertices"], pred["cam_t"], prima_camera)

    if args.baseline_only:
        clean = (~df["truncated"] & ~df["overlapping"]).to_numpy()
        summary["clean"] = {"n": int(clean.sum()), **{c: float(base_df.loc[clean, c].median()) for c in FIT_COLUMNS}}
        summary["proportions"] = {"prima": {k: float(df[k].median()) for k in ("neck_to_body", "leg_to_body")}}
        pd.concat([df, base_df], axis=1).to_csv(out / "metrics.csv", index=False)
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        if not args.no_videos:
            write_visuals(source, out, df, faces, pred["vertices"], pred["cam_t"])
        print(json.dumps(summary, indent=2))
        print(f"Results in {out}")
        return

    # Measure the body on clean views, on animals that PRIMA at least placed correctly.
    clean = (~df["truncated"] & ~df["overlapping"]).to_numpy()
    # Specks a few pixels tall say almost nothing about shape, and may not even be the same species; in Camel_1
    # they outnumbered the camel two to one. They are still posed and drawn, just not used to measure the body.
    usable = clean & (base_df["top_recall"].to_numpy() >= 0.3) & (df["box_h"].to_numpy() >= MIN_SHAPE_BOX_H)
    if not usable.any():
        usable = clean & (base_df["top_recall"].to_numpy() >= 0.3)
    if not usable.any():
        usable = clean if clean.any() else np.ones(len(df), dtype=bool)
    side_on = np.abs(pred["global_orient"][:, 2, 0]) < SIDE_ON_LIMIT
    train, held_out_tracks = choose_shape_instances(usable, side_on, df["track_id"].to_numpy())
    if not len(train):
        train = np.where(usable)[0]
        held_out_tracks = []
    print(f"Measuring the body on {len(train)} of {len(df)} instances; tracks held out to test it: {held_out_tracks}")

    smal = load_smal(args.prima_dir)
    betas = pred["betas"][train].mean(axis=0)

    inst_all = build_instances(pred, df, masks, source, device)
    if source.has_lidar and args.depth_weight > 0:
        attach_lidar(inst_all, source, df, masks)
    inst_train = inst_all.subset(train)
    search_inst = inst_train.subset(np.linspace(0, len(train) - 1, min(len(train), 96)).astype(int))
    print("Searching body plans")
    horn_mode, species_info = choose_horns(source, df, train, args.horns)
    humps = species_info["humps"] if args.humps == "auto" else int(args.humps)
    species_info["humps"] = humps
    print(f"Horns: {horn_mode} ({species_info.get('reason', 'set on the command line')}); humps: {humps}")
    named = {"auto": species_info.get("ears"), "masks": None, "smal": 1.0}
    ears = named[args.ears] if args.ears in named else float(args.ears)
    species_info["ears"] = ears
    bounds = dict(BODY_BOUNDS)
    if ears is not None:  # a known ear size, held fixed: masks barely show ears, so fitted ones only pad the outline
        thickness = 1.0 if args.ears == "smal" else EAR_THICKNESS * ears
        bounds["ears"] = (ears, ears, thickness, thickness)
        print(f"Ears: {ears} of SMAL's length, fixed")
    make_body = lambda init, horns=None: AnimalBody(**smal, betas=betas, init=init, bounds=bounds, horns=horns, humps=humps).to(device)
    if args.ablation == "none":
        best_init, best_horns, search, horn_decision = search_body(make_body, search_inst, horns=horn_mode)
    else:  # no body-plan search: PRIMA's own proportions (horns and humps only as forced on the command line)
        forced = horn_mode if horn_mode in ("paired", "nasal_two", "nasal_one") else None
        from animal3d.fit import horn_starts
        best_init, best_horns, search = make_init(), (next(iter(horn_starts(forced).values())) if forced else None), []
        horn_decision = {"mode": horn_mode, "ablation": args.ablation}
    print(f"Best start: neck {best_init['neck'][0]}, legs {best_init['fore_upper'][0]}, thickness {best_init['neck'][1]}, horns {best_horns}")
    body = make_body(best_init, best_horns)
    if horn_decision.get("horn_shape") == "species prior, not measured":
        body.horns.raw.requires_grad_(False)
    # The body shape comes from the masks alone. With LiDAR in this stage too, the shared shape absorbed the strain
    # of noisy depth on a few points per animal: Rhino_1's neck grew from 0.41 to 1.16 body lengths.
    if args.ablation in ("smal", "betas"):
        poses_shape = Poses(inst_train).to(device)
        print("Ablation: settling pose on the masks, SMAL shape " + ("coefficients fitted" if args.ablation == "betas" else "fixed"))
        optimize(body, poses_shape, 1500, args.batch, lr_pose=0.01)
        if args.ablation == "betas":
            optimize(body, poses_shape, args.bone_iters, args.batch, lr_pose=0.01, lr_betas=0.02, w_betas=1e-5)
        for parameter in body.parameters():
            parameter.requires_grad_(False)
    else:
        fit_body(body, inst_train, args.bone_iters, args.offset_iters, args.batch)
    scales = {n: [round(float(v), 2) for v in body.scales()[i]] for i, n in enumerate(body.group_names)}
    print("Fitted (length, thickness):", scales)

    print("Fitting every instance with the body fixed")
    poses = Poses(inst_all).to(device)
    # Enough passes for each instance to be visited about 300 times.
    pose_iters = int(np.clip(len(df) * 300 / args.batch, 1000, 12000))
    optimize(body, poses, pose_iters, args.batch, lr_pose=0.01, w_temporal=0.02 if source.sequence else 0.0, w_depth=args.depth_weight)

    vertices = posed_vertices(body, poses)
    with torch.no_grad():
        all_idx = torch.arange(len(df), device=device)
        cam_t = poses.translation(all_idx).cpu().numpy()
        orient, pose = (m.cpu().numpy() for m in poses.matrices(all_idx))
        rest = body.rest()[0].cpu().numpy()

    camera_for = lambda w, h: fit_camera(source, w, h)
    print("Scoring the fitted body")
    fit_df = score_meshes(source, df, masks, body.faces, vertices, cam_t, camera_for)

    held_out = df["track_id"].isin(held_out_tracks).to_numpy()
    summary["shape"] = {
        "measured_on_instances": int(len(train)), "held_out_tracks": held_out_tracks,
        "species": species_info, "horn_decision": horn_decision,
        "search": search, "chosen_start": {"neck": best_init["neck"][0], "leg": best_init["fore_upper"][0], "thickness": best_init["neck"][1]},
        "bone_scales_length_thickness": scales,
        "horns": asdict(body.horns.shape()) if body.horns is not None else None,
        "humps": body.humps.describe() if body.humps is not None else None,
        "proportions": {"prima": {k: float(df[k].median()) for k in ("neck_to_body", "leg_to_body")}, "fitted": body.proportions()},
    }
    summary["clean_all"] = medians(base_df, fit_df, clean)
    summary["clean_shape_tracks"] = medians(base_df, fit_df, clean & ~held_out)
    if held_out.any():
        summary["clean_held_out_tracks"] = medians(base_df, fit_df, clean & held_out)
    summary["iou_below_0.5_of_all"] = {"prima": int((base_df["iou"] < 0.5).sum()), "fitted": int((fit_df["iou"] < 0.5).sum()), "n": int(len(df))}

    # Metric scale and the independent check against LiDAR.
    metres_per_unit = None
    columns = pd.concat([df, base_df.add_prefix("prima_"), fit_df], axis=1)
    if source.has_lidar:
        print("Comparing PRIMA's body in the same camera with the LiDAR")
        moved = np.stack([
            move_to_camera(pred["cam_t"][i], PRIMA_FOCAL_RATIO * max(w, h), w, h, camera_for(w, h))[0]
            for i, (w, h) in enumerate(zip(df["width"], df["height"]))
        ])
        base_lidar = score_meshes(source, df, masks, faces, pred["vertices"], moved, camera_for)
        columns["prima_relief_error_m"] = base_lidar["relief_error_m"]
        columns["prima_relief_error_held_m"] = base_lidar["relief_error_held_m"]
        tracks = track_scales(columns, clean)
        lidar_seen = columns.loc[clean, "relief_error_m"].notna()
        summary["lidar"] = {
            "clean_instances_with_lidar": int(lidar_seen.sum()), "of_clean": int(clean.sum()),
            "points_per_instance_median": float(columns.loc[clean, "lidar_on_mesh"].median()),
            "relief_error_m": {"prima": float(columns.loc[clean, "prima_relief_error_m"].median()),
                               "fitted": float(columns.loc[clean, "relief_error_m"].median()),
                               "flat": float(columns.loc[clean, "flat_relief_m"].median())},
            "used_in_fit": bool(args.depth_weight > 0),
            # The fair check when LiDAR is used in the fit: only the held-out half of its points.
            "relief_error_held_m": {"prima": float(columns.loc[clean, "prima_relief_error_held_m"].median()),
                                    "fitted": float(columns.loc[clean, "relief_error_held_m"].median()),
                                    "flat": float(columns.loc[clean, "flat_relief_held_m"].median())},
            "tracks": tracks.to_dict("records"),
        }
        if len(tracks):
            metres_per_unit = float(tracks["metres_per_unit"].median())
            summary["lidar"]["metres_per_unit"] = metres_per_unit
    columns.to_csv(out / "metrics.csv", index=False)

    reliable = clean & (fit_df["iou"].to_numpy() >= 0.6)
    renderers = Renderers(body.faces)
    colors = bake_colors(source, vertices, cam_t, df, fit_df, masks, renderers, camera_for, body.mirror.cpu().numpy())
    sizes = write_model(out, body, pose, reliable, colors, metres_per_unit)
    summary["model_size"] = {"unit": "metres" if metres_per_unit else "model units (no LiDAR, so size is unknown)", **sizes}
    if metres_per_unit and summary["lidar"]["tracks"]:
        base_units = {k: v / metres_per_unit for k, v in sizes.items()}
        for track in summary["lidar"]["tracks"]:
            track["shoulder_height_m"] = base_units["shoulder_height"] * track["metres_per_unit"]
            track["top_height_m"] = base_units["top_height"] * track["metres_per_unit"]
            track["length_m"] = base_units["length"] * track["metres_per_unit"]

    np.savez_compressed(out / "fit.npz", faces=body.faces, rest_vertices=rest, vertices=vertices.astype(np.float32),
                        cam_t=cam_t, global_orient=orient, pose=pose, frame=df["frame"].to_numpy(), track_id=df["track_id"].to_numpy(),
                        log_scale=body.log_scale.detach().cpu().numpy(), betas=body.betas.detach().cpu().numpy(),
                        offsets=body.symmetric_offsets().detach().cpu().numpy())
    if args.mesh_stride:
        (out / "meshes").mkdir(exist_ok=True)
        stride = args.mesh_stride if source.sequence else 1
        for row in np.where(df["frame"] % stride == 0)[0]:
            export_mesh(out / "meshes" / f"frame{df.loc[row, 'frame']:05d}_animal{df.loc[row, 'track_id']:02d}.obj",
                        vertices[row], cam_t[row], body.faces, track_color(int(df.loc[row, "track_id"])))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "shape"} | {"proportions": summary["shape"]["proportions"]}, indent=2))

    plot_df = pd.concat([df[["box_h", "truncated", "overlapping"]], fit_df], axis=1)
    if len(plot_df) > 5:
        plot_summary(plot_df, str(out / "fit_vs_size.png"))
    if not args.no_videos:
        half_extent = float(np.ptp(rest, axis=0).max()) * 0.62
        write_visuals(source, out, df, faces, pred["vertices"], pred["cam_t"], vertices, cam_t, orient, fit_df, body.faces, half_extent)
    print(f"Results in {out}")


if __name__ == "__main__":
    main()
