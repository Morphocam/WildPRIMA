#!/usr/bin/env python
"""Build a giraffe 3D model from a video, starting from the results of run.py.

PRIMA's single SMAL template cannot produce giraffe proportions, so this fits a giraffe body (shared
by all animals) and per-frame poses to the giraffes' instance masks. Requires outputs/<video>/ from
run.py. Writes to outputs/<video>_giraffe/.

    python fit_giraffe.py 1.mp4
    python fit_giraffe.py 1.mp4 --preview-only   # fit and score only, without the videos
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from giraffe3d.body import GIRAFFE_BOUNDS, GIRAFFE_INIT, GiraffeBody
from giraffe3d.detection import GiraffeDetector
from giraffe3d.fit import Instances, Poses, optimize, posed_vertices, prepare_mask
from giraffe3d.metrics import fit_metrics
from giraffe3d.render import FrameRenderer, ProfileRenderer, export_mesh, track_color, turntable_frames
from run import PANEL, PROFILE_VIEW, ROOT, SLOTS, SlotMap, build_dashboard, label, square_crop, transcode_h264

FOCAL_LENGTH = 1000.0
IMAGE_SIZE = 256.0
PROFILE_HALF_EXTENT = 1.7
# Tracks kept out of the shape fit, to test the giraffe model on animals it has not seen.
HOLDOUT_TRACKS = (6, 7, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--prima-dir", type=Path, default=ROOT / "third_party/PRIMA")
    parser.add_argument("--bone-iters", type=int, default=2000)
    parser.add_argument("--offset-iters", type=int, default=2000)
    parser.add_argument("--pose-iters", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--preview-only", action="store_true")
    return parser.parse_args()


def load_masks(video: Path, df: pd.DataFrame, cache: Path, det_thresh: float = 0.7) -> list:
    """Per row of metrics.csv: its instance mask cropped to its bounding box, plus the crop's origin."""
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    detector = GiraffeDetector(det_thresh)
    cap = cv2.VideoCapture(str(video))
    masks: list = [None] * len(df)
    rows_by_frame = {frame: group for frame, group in df.groupby("frame")}
    for frame_idx in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), desc="masks"):
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx not in rows_by_frame:
            continue
        dets, _ = detector(frame)
        for row_idx, row in rows_by_frame[frame_idx].iterrows():
            box = row[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32)
            error = np.abs(dets.boxes - box).sum(axis=1)
            best = int(error.argmin())
            if error[best] > 1.0:
                raise RuntimeError(f"Detections differ from metrics.csv at frame {frame_idx}; re-run run.py")
            ys, xs = np.nonzero(dets.masks[best])
            y0, x0 = int(ys.min()), int(xs.min())
            masks[row_idx] = (dets.masks[best][y0 : ys.max() + 1, x0 : xs.max() + 1].copy(), (y0, x0))
    cap.release()
    cache.write_bytes(pickle.dumps(masks))
    return masks


def full_mask(entry, height: int, width: int) -> np.ndarray:
    crop, (y0, x0) = entry
    mask = np.zeros((height, width), dtype=bool)
    mask[y0 : y0 + crop.shape[0], x0 : x0 + crop.shape[1]] = crop
    return mask


def build_instances(pred, df, masks, keep: np.ndarray, height: int, width: int, device) -> Instances:
    rng = np.random.default_rng(0)
    prepared = [
        prepare_mask(full_mask(masks[i], height, width), df.loc[i, ["x1", "y1", "x2", "y2"]].to_numpy(np.float32), rng)
        for i in keep
    ]
    as_tensor = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
    return Instances(
        r0=as_tensor(pred["global_orient"][keep]),
        p0=as_tensor(pred["pose"][keep]),
        t0=as_tensor(pred["cam_t"][keep]),
        center=as_tensor([p[0] for p in prepared]),
        size=as_tensor([p[1] for p in prepared]),
        distance=as_tensor([p[2] for p in prepared])[:, None],
        samples=as_tensor([p[3] for p in prepared]),
        track=pred["track_id"][keep],
        frame=pred["frame"][keep],
    )


def load_body(prima_dir: Path, betas: np.ndarray, device) -> GiraffeBody:
    from chumpy.ch import materialize

    with open(prima_dir / "data/smal/my_smpl_00781_4_all.pkl", "rb") as f:
        smal = pickle.load(f, encoding="latin1")
    parents = np.array(smal["kintree_table"][0]).astype(np.int64)
    parents[parents > 1000] = -1  # the root's parent is stored as an unsigned -1
    body = GiraffeBody(
        materialize(smal["v_template"]), materialize(smal["shapedirs"])[:, :, :41], smal["J_regressor"].toarray(),
        parents, materialize(smal["weights"]), materialize(smal["f"], dtype=np.int32), betas, init=GIRAFFE_INIT, bounds=GIRAFFE_BOUNDS,
    )
    return body.to(device)


def score(vertices, cam_t, masks, renderer, height, width) -> pd.DataFrame:
    rows = []
    for i in tqdm(range(len(masks)), desc="scoring"):
        silhouette = renderer.silhouette(vertices[i], cam_t[i])
        rows.append(fit_metrics(silhouette, full_mask(masks[i], height, width)))
    return pd.DataFrame(rows)


def medians(fit: pd.DataFrame, base: pd.DataFrame, where: np.ndarray) -> dict:
    columns = ("iou", "height_ratio", "width_ratio", "area_ratio", "neck_recall")
    return {
        "n": int(where.sum()),
        **{c: {"prima": float(base.loc[where, c].median()), "giraffe": float(fit.loc[where, c].median())} for c in columns},
    }


def mean_rotations(matrices: np.ndarray) -> np.ndarray:
    """Chordal mean over the first axis, for rotation matrices of shape (N, ..., 3, 3)."""
    u, _, vt = np.linalg.svd(matrices.mean(axis=0))
    sign = np.sign(np.linalg.det(u @ vt))
    u[..., :, 2] *= sign[..., None]
    return u @ vt


def bake_colors(video, vertices, cam_t, df, fit_df, masks, renderer, mirror, height, width, min_iou=0.6) -> np.ndarray:
    """Per-vertex color (0..1): median over frames where the vertex is visible and inside the mask."""
    cap = cv2.VideoCapture(str(video))
    good = df.index[fit_df["iou"] >= min_iou].to_numpy()
    by_frame = {f: g for f, g in df.loc[good].groupby("frame")}
    samples = np.full((len(good), vertices.shape[1], 3), np.nan, dtype=np.float32)
    slot = {row: i for i, row in enumerate(good)}
    principal = np.array([width / 2, height / 2])
    focal = renderer.camera.fx
    for frame_idx in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), desc="colors"):
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx not in by_frame:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        for row in by_frame[frame_idx].index:
            camera = vertices[row] + cam_t[row]
            uv = focal * camera[:, :2] / camera[:, 2:] + principal
            px = np.clip(np.round(uv).astype(int), 0, [width - 1, height - 1])
            depth = renderer.depth(vertices[row], cam_t[row])[px[:, 1], px[:, 0]]
            mask = full_mask(masks[row], height, width)[px[:, 1], px[:, 0]]
            visible = mask & (depth > 0) & (np.abs(depth - camera[:, 2]) < 0.05)
            samples[slot[row], visible] = rgb[px[visible, 1], px[visible, 0]]
    cap.release()
    colors = np.nanmedian(samples, axis=0)
    colors = np.where(np.isnan(colors), colors[mirror], colors)  # a side never seen borrows the other's
    colors = np.where(np.isnan(colors), np.nanmedian(colors, axis=0), colors)
    return np.clip(colors, 0, 1)


def write_model(out, body, pose, orient, fit_df, clean, video, vertices, cam_t, df, masks, renderer, height, width) -> None:
    """The standalone model: mean pose over well-fitted frames, with colors from the video."""
    import trimesh

    device = body.log_scale.device
    reliable = clean & (fit_df["iou"].to_numpy() >= 0.6)
    standing_pose = mean_rotations(pose[reliable])
    with torch.no_grad():
        standing = body(torch.eye(3, device=device)[None], torch.as_tensor(standing_pose, device=device, dtype=torch.float32)[None])[0].cpu().numpy()
    colors = bake_colors(video, vertices, cam_t, df, fit_df, masks, renderer, body.mirror.cpu().numpy(), height, width)

    # z-up body frame to y-up, as glTF viewers expect.
    y_up = standing @ PROFILE_VIEW.T
    mesh = trimesh.Trimesh(y_up, body.faces, vertex_colors=np.c_[colors, np.ones(len(colors))], process=False)
    mesh.export(out / "giraffe_model.glb")
    mesh.export(out / "giraffe_model.obj")
    print("Model extent, x (tail-nose), y (up), z (side):", np.round(y_up.max(axis=0) - y_up.min(axis=0), 2))

    raw = out / "_turntable.mp4"
    writer = None
    for frame in turntable_frames(standing, body.faces, colors):
        if writer is None:
            writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 24, (frame.shape[1], frame.shape[0]))
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    transcode_h264(raw, out / "giraffe_turntable.mp4", 24)


def write_videos(video, out, df, base_vertices, base_t, fit_vertices, fit_t, fit_orient, fit_df, faces, focal, fps):
    cap = cv2.VideoCapture(str(video))
    width, height = int(cap.get(3)), int(cap.get(4))
    frame_renderer = FrameRenderer(width, height, focal, faces)
    profile = ProfileRenderer(PANEL, faces, PROFILE_HALF_EXTENT, PROFILE_VIEW)
    slots = SlotMap(SLOTS)
    raw_overlay, raw_compare = out / "_overlay.mp4", out / "_comparison.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    overlay_writer = cv2.VideoWriter(str(raw_overlay), fourcc, fps, (width, height))
    compare_writer = None
    groups = {frame: group.index.to_numpy() for frame, group in df.groupby("frame")}

    for frame_idx in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), desc="video"):
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rows = groups.get(frame_idx, np.array([], dtype=int))
        tracks = df.loc[rows, "track_id"].to_numpy()
        colors = [track_color(int(t)) for t in tracks]
        base = frame_renderer.overlay(frame, base_vertices[rows], base_t[rows], colors)
        fitted = frame_renderer.overlay(frame, fit_vertices[rows], fit_t[rows], colors)
        assigned = slots.update(tracks)
        panels = {}
        for row, track, color in zip(rows, tracks, colors):
            if int(track) not in assigned:
                continue
            box = df.loc[row, ["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
            crop = square_crop(fitted, box, PANEL)
            label(crop, f"#{track} IoU {fit_df.loc[row, 'iou']:.2f}", (8, 20))
            view = profile(fit_vertices[row], fit_orient[row], color).copy()
            label(view, f"#{track} shape, fixed scale", (8, 20))
            panels[assigned[int(track)]] = (crop, view)
        overlay_writer.write(cv2.cvtColor(fitted, cv2.COLOR_RGB2BGR))
        canvas = cv2.cvtColor(build_dashboard(base, fitted, panels, ("PRIMA as-is", "Fitted giraffe model")), cv2.COLOR_RGB2BGR)
        if compare_writer is None:
            compare_writer = cv2.VideoWriter(str(raw_compare), fourcc, fps, (canvas.shape[1], canvas.shape[0]))
        compare_writer.write(canvas)
    cap.release()
    overlay_writer.release()
    compare_writer.release()
    transcode_h264(raw_overlay, out / "overlay.mp4", fps)
    transcode_h264(raw_compare, out / "comparison.mp4", fps)


def main() -> None:
    args = parse_args()
    source = ROOT / "outputs" / args.video.stem
    out = ROOT / "outputs" / f"{args.video.stem}_giraffe"
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    df = pd.read_csv(source / "metrics.csv")
    pred = dict(np.load(source / "predictions.npz"))
    cap = cv2.VideoCapture(str(args.video))
    width, height, fps = int(cap.get(3)), int(cap.get(4)), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    focal = FOCAL_LENGTH / IMAGE_SIZE * max(width, height)
    image_size = (width, height)
    masks = load_masks(args.video, df, out / "masks.pkl")

    usable = (~df["truncated"] & ~df["overlapping"] & (df["neck_recall"] >= 0.3)).to_numpy()
    held_out = df["track_id"].isin(HOLDOUT_TRACKS).to_numpy()
    train = np.where(usable & ~held_out)[0]
    print(f"Shape fit on {len(train)} of {len(df)} instances; held out tracks {HOLDOUT_TRACKS} ({held_out.sum()} instances)")

    body = load_body(args.prima_dir, pred["betas"].mean(axis=0), device)
    inst = build_instances(pred, df, masks, train, height, width, device)
    poses = Poses(inst).to(device)
    print("Stage 1: pose and camera only, to settle the stretched skeleton on the masks")
    optimize(body, poses, focal, image_size, 1500, args.batch, lr_pose=0.01)
    print("Stage 2: bone scales")
    optimize(body, poses, focal, image_size, args.bone_iters, args.batch, lr_pose=0.01, lr_shape=0.003)
    print("Stage 3: vertex offsets")
    optimize(body, poses, focal, image_size, args.offset_iters, args.batch, lr_pose=0.005, lr_shape=0.001,
             lr_offsets=0.0005, w_laplacian=40.0, w_lateral=0.5)
    scales = {n: [round(float(v), 2) for v in body.scales()[i]] for i, n in enumerate(body.group_names)}
    print("Fitted (length, thickness):", scales)

    print("Stage 4: all instances with the shape frozen")
    for parameter in body.parameters():
        parameter.requires_grad_(False)
    everything = np.arange(len(df))
    inst_all = build_instances(pred, df, masks, everything, height, width, device)
    poses_all = Poses(inst_all).to(device)
    optimize(body, poses_all, focal, image_size, args.pose_iters, args.batch, lr_pose=0.01, w_temporal=0.02)

    vertices = posed_vertices(body, poses_all)
    with torch.no_grad():
        all_idx = torch.arange(len(df), device=device)
        cam_t = poses_all.translation(all_idx).cpu().numpy()
        orient, pose = (m.cpu().numpy() for m in poses_all.matrices(all_idx))
        rest = body.rest()[0].cpu().numpy()

    renderer = FrameRenderer(width, height, focal, body.faces)
    fit_df = score(vertices, cam_t, masks, renderer, height, width)
    base_df = df[list(fit_df.columns)]
    clean = (~df["truncated"] & ~df["overlapping"]).to_numpy()
    summary = {
        "proportions": {"prima": {k: float(df[k].median()) for k in ("neck_to_body", "leg_to_body")}, "giraffe": body.proportions()},
        "bone_scales_length_thickness": scales,
        "clean_all": medians(fit_df, base_df, clean),
        "clean_shape_fit_tracks": medians(fit_df, base_df, clean & ~held_out),
        "clean_held_out_tracks": medians(fit_df, base_df, clean & held_out),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    pd.concat([df[["frame", "track_id", "x1", "y1", "x2", "y2", "truncated", "overlapping"]], fit_df], axis=1).to_csv(out / "metrics.csv", index=False)

    write_model(out, body, pose, orient, fit_df, clean, args.video, vertices, cam_t, df, masks, renderer, height, width)
    np.savez_compressed(out / "fit.npz", faces=body.faces, rest_vertices=rest, vertices=vertices.astype(np.float32),
                        cam_t=cam_t, global_orient=orient, pose=pose, frame=pred["frame"], track_id=pred["track_id"],
                        log_scale=body.log_scale.detach().cpu().numpy(), betas=body.betas.detach().cpu().numpy(),
                        offsets=body.symmetric_offsets().detach().cpu().numpy())
    for row in np.where(df["frame"] % 10 == 0)[0]:
        export_mesh(out / "meshes" / f"frame{df.loc[row, 'frame']:05d}_giraffe{df.loc[row, 'track_id']:02d}.obj",
                    vertices[row], cam_t[row], body.faces, track_color(int(df.loc[row, "track_id"])))

    if not args.preview_only:
        write_videos(args.video, out, df, pred["vertices"], pred["cam_t"], vertices, cam_t, orient, fit_df, body.faces, focal, fps)
    print(f"Results in {out}")


if __name__ == "__main__":
    main()
