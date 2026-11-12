"""Step 4d: test-time refinement of the student (and, for a fair comparison, of PRIMA) on the video's masks only.

    python student_refine.py [--visits 50] [--split val|test|all] [RUN ...]

The student's per-frame prediction is turned into one body per animal track (median bone scales and betas, median
metres per unit), then pose, orientation and translation of every frame are refined against that frame's detected
mask with the per-clip fit's own optimiser (animal3d.fit.optimize): mask terms, pose prior, temporal smoothing. No
LiDAR, no body-plan search, no offsets or parts, and the bone scales stay as the student predicted them. PRIMA gets
exactly the same treatment (one body per track from its betas, unit bone scales) so that the comparison isolates the
starting point. Distance in metres = refined depth x the student's metres per unit; PRIMA has no metric scale.

Writes new_output/eval_refine/<run>/metrics_<init>.csv and new_output/eval_refine/summary.csv / summary.json.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from animal3d.body import BODY_BOUNDS, AnimalBody
from animal3d.fit import IndexedAdam, Instances, Poses, mask_loss, pose_prior, posed_vertices, project_to_crop, temporal_loss
from pose_prior import NEW, ROOT, run_folders, split_of
from student import CACHE, Data, Student, camera_translation, load_prima

OUT = NEW / "eval_refine"


def refine(body, poses: Poses, iters: int, batch: int, w_relief: float, w_orient: float, lr=0.01, w_pose=1e-2,
           w_temporal=0.02, seed=0) -> None:
    """animal3d.fit.optimize's pose stage (mask terms, pose prior, temporal term, same learning rates), plus a pull of the
    mesh's depth relief (camera depth about its own mean, per vertex) towards the network's prediction. Masks cannot see
    relief; the starting prediction can, for the student because it was trained with the LiDAR relief term."""
    inst = poses.inst
    with torch.no_grad():
        all_idx = torch.arange(len(inst), device=inst.r0.device)
        z0 = torch.cat([(body(*poses.matrices(i)) + poses.translation(i)[:, None])[..., 2] for i in all_idx.split(64)])
        relief0 = z0 - z0.mean(dim=1, keepdim=True)
    opt = IndexedAdam([(poses.d_orient, lr), (poses.d_pose, lr), (poses.d_cam, lr * 0.1)])
    generator = torch.Generator(device=inst.r0.device).manual_seed(seed)
    for _ in range(iters):
        idx = torch.randperm(len(inst), generator=generator, device=inst.r0.device)[:batch]
        vertices = body(*poses.matrices(idx))
        t = poses.translation(idx)
        uv = project_to_crop(vertices, t, inst.camera[idx], inst.center[idx], inst.size[idx])
        outside, coverage = mask_loss(uv, inst.distance[idx], inst.samples[idx])
        loss = outside + coverage + w_pose * pose_prior(poses, idx) + w_temporal * temporal_loss(poses)
        if w_relief:
            z = (vertices + t[:, None])[..., 2]
            loss = loss + w_relief * (z - z.mean(dim=1, keepdim=True) - relief0[idx]).pow(2).mean()
        if w_orient:
            loss = loss + w_orient * poses.d_orient[idx].pow(2).sum(dim=-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step(idx)


def instances(m, r0, p0, t0, rows) -> Instances:
    """animal3d.fit.Instances for the given rows of a cached run (masks, cameras), starting from (r0, p0, t0)."""
    t = torch.as_tensor(rows, device=r0.device)
    return Instances(r0=r0[t], p0=p0[t], t0=t0[t], camera=m["camera"][t], center=m["center"][t], size=m["size"][t],
                     distance=m["distance"][t], samples=m["samples"][t],
                     track=m["track"][t].cpu().numpy(), frame=m["frame"][t].cpu().numpy())


def refine_run(name, student, smal, device, visits, w_orient=0.0, w_relief=0.0, batch=48):
    import run as R
    from animal3d.sources import open_source

    out, inp, calib = run_folders()[name]
    data = Data([name], device, usable_only=False)
    m = data.meta
    n = data.n
    with torch.no_grad():
        o = {k: [] for k in ("orient", "pose", "betas", "log_scale", "log_mpu", "t")}
        for s in range(0, n, 64):
            idx = torch.arange(s, min(s + 64, n), device=device)
            r = student(data.tokens[idx.cpu()].to(device).float())
            r["t"] = camera_translation(r["cam"], m, idx)
            for k in o:
                o[k].append(r[k])
        o = {k: torch.cat(v) for k, v in o.items()}
    df, pred, masks, _ = R.load_prima_results(out)
    source = open_source(inp, None, calib)
    cam = lambda w, h: R.fit_camera(source, w, h)
    prima_t = torch.as_tensor(np.stack([
        R.move_to_camera(pred["cam_t"][i], R.PRIMA_FOCAL_RATIO * max(w, h), w, h, cam(w, h))[0]
        for i, (w, h) in enumerate(zip(df.width, df.height))]), dtype=torch.float32, device=device)
    f32 = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
    starts = {
        "student": dict(r0=o["orient"], p0=o["pose"], t0=o["t"], betas=o["betas"], log_scale=o["log_scale"]),
        "prima": dict(r0=f32(pred["global_orient"]), p0=f32(pred["pose"]), t0=prima_t, betas=f32(pred["betas"]), log_scale=None),
    }
    tracks = m["track"].cpu().numpy()
    faces = None
    results = {}
    for init, st in starts.items():
        t0 = time.time()
        verts = np.zeros((n, 3889, 3), np.float32)
        trans = np.zeros((n, 3), np.float32)
        for track in np.unique(tracks):
            rows = np.where(tracks == track)[0]
            betas = st["betas"][rows].median(dim=0).values
            body = AnimalBody(**smal, betas=betas.cpu().numpy(), bounds=BODY_BOUNDS).to(device)
            with torch.no_grad():
                if st["log_scale"] is not None:
                    body.log_scale.copy_(st["log_scale"][rows].median(dim=0).values)
            for p in body.parameters():
                p.requires_grad_(False)
            faces = body.faces if hasattr(body, "faces") else faces
            inst = instances(m, st["r0"], st["p0"], st["t0"], rows)
            poses = Poses(inst).to(device)
            if visits > 0:
                iters = int(max(len(rows) * visits / batch, 100))
                refine(body, poses, iters, min(batch, len(rows)), w_relief=w_relief, w_orient=w_orient)
            v = posed_vertices(body, poses)
            verts[rows] = v[:, :3889]
            with torch.no_grad():
                trans[rows] = poses.translation(torch.arange(len(rows), device=device)).cpu().numpy()
        seconds = time.time() - t0
        faces_np = np.asarray(smal["faces"], dtype=np.int64)
        scored = R.score_meshes(source, df, masks, faces_np, verts, trans, cam)
        if init == "student":
            mpu = o["log_mpu"].exp().cpu().numpy()
            track_mpu = pd.Series(mpu).groupby(tracks).transform("median").to_numpy()
            scored["distance_m"] = trans[:, 2] * track_mpu
        results[init] = (scored, seconds)
    metrics = pd.read_csv(out / "metrics.csv")
    target = OUT / name
    target.mkdir(parents=True, exist_ok=True)
    clean = (~metrics.truncated & ~metrics.overlapping).to_numpy()
    depth = metrics["lidar_depth_m"].to_numpy()
    row = {"run": name, "split": split_of(name), "instances": n, "clean": int(clean.sum()), "visits": visits, "w_orient": w_orient, "w_relief": w_relief}
    med = lambda frame, col: float(np.nanmedian(frame.loc[clean, col])) if col in frame and clean.any() else None
    for init, (scored, seconds) in results.items():
        scored.to_csv(target / f"metrics_{init}_v{visits}_o{w_orient:g}_r{w_relief:g}.csv", index=False)
        row[f"iou_{init}_refined"] = med(scored, "iou")
        row[f"lidar_held_{init}_refined"] = med(scored, "relief_error_held_m")
        row[f"seconds_{init}"] = seconds
    ok = clean & np.isfinite(depth) & (depth > 0)
    d = results["student"][0]["distance_m"].to_numpy()
    row["distance_rel_error_student_refined"] = float(np.nanmedian(np.abs(d[ok] - depth[ok]) / depth[ok])) if ok.any() else None
    return row


CHOSEN_RELIEF = 10.0  # chosen on val: lowest held-out LiDAR error of the refined student (sweep in eval_refine/summary.csv)


def paper_table() -> None:
    """paper/generated/tab_refine.tex: medians per split, refinement at 50 visits, no orientation pull."""
    s = pd.read_csv(OUT / "summary.csv")
    s[["w_orient", "w_relief"]] = s[["w_orient", "w_relief"]].fillna(0.0)
    s = s[(s.visits == 50) & (s.w_orient == 0)]
    base = pd.read_csv(NEW / "eval" / "summary.csv")
    f = lambda x, d: "--" if x is None or not np.isfinite(x) else f"{x:.{d}f}"
    lines = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
             r"& \multicolumn{3}{c}{Validation (8 windows)} & \multicolumn{3}{c}{Test (6 windows)} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Method & IoU $\uparrow$ & LiDAR (m) $\downarrow$ & Dist. err $\downarrow$ & IoU $\uparrow$ & LiDAR (m) $\downarrow$ & Dist. err $\downarrow$ \\",
             r"\midrule"]
    def cells(split, iou, lidar, dist, frame):
        g = frame[frame.split == split]
        med = lambda c: float(g[c].median()) if c and c in g and g[c].notna().any() else None
        d = med(dist)
        return [f(med(iou), 3), f(med(lidar), 3), "--" if d is None else f"{100 * d:.1f}\\%"]
    animer_path = NEW / "baseline_animer" / "summary.csv"  # baseline_animer.py
    animer = pd.read_csv(animer_path).rename(columns={"set": "split"}) if animer_path.exists() else None
    rows = ([("AniMer \\cite{lyu2025animer}", animer, "iou_animer", "lidar_held_animer", None)] if animer is not None else []) + [
            ("PRIMA \\cite{yu_prima}", base, "iou_prima", "lidar_held_prima", None),
            ("Student", base, "iou_student", "lidar_held_student", "distance_rel_error_student")]
    breaks = (len(rows), len(rows) + 2)
    for w, label in ((CHOSEN_RELIEF, "relief anchor"), (0.0, "no anchor")):
        r = s[s.w_relief == w]
        rows += [(f"PRIMA + refinement ({label})", r, "iou_prima_refined", "lidar_held_prima_refined", None),
                 (f"Student + refinement ({label})", r, "iou_student_refined", "lidar_held_student_refined", "distance_rel_error_student_refined")]
    for i, (label, frame, iou, lidar, dist) in enumerate(rows):
        if i in breaks:
            lines.append(r"\midrule")
        lines.append(label + " & " + " & ".join(cells("val", iou, lidar, dist, frame) + cells("test", iou, lidar, dist, frame)) + r" \\")
    lines.append(r"\midrule")
    lines.append("Per-clip fit (video + LiDAR) & " + " & ".join(cells("val", "iou_fit", "lidar_held_fit", None, base) + cells("test", "iou_fit", "lidar_held_fit", None, base)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = ROOT / "paper" / "generated" / "tab_refine.tex"
    out.write_text("% made by student_refine.py --table\n" + "\n".join(lines) + "\n")
    print(out.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*")
    parser.add_argument("--visits", type=int, default=50, help="refinement steps per instance (0 = no refinement)")
    parser.add_argument("--split", default="val", choices=["val", "test", "all"])
    parser.add_argument("--w-orient", type=float, default=0.0, help="pull of the orientation towards its start")
    parser.add_argument("--w-relief", type=float, default=0.0, help="pull of the depth relief towards the prediction's")
    parser.add_argument("--table", action="store_true", help="only write paper/generated/tab_refine.tex")
    args = parser.parse_args()
    if args.table:
        return paper_table()
    device = "cuda"
    import run as R

    smal = R.load_smal(ROOT / "third_party/PRIMA")
    prima = load_prima(device)
    student = Student(prima.model.smal_head).to(device)
    del prima
    student.load_state_dict(torch.load(NEW / "student" / "student.pt", map_location=device)["state"])
    student.eval()
    cached = sorted(p.name for p in CACHE.iterdir() if (p / "meta.npz").exists())
    splits = ("val", "test") if args.split == "all" else (args.split,)
    names = args.runs or [n for n in cached if split_of(n) in splits]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        rows.append(refine_run(name, student, smal, device, args.visits, args.w_orient, args.w_relief))
        print(json.dumps(rows[-1]), flush=True)
    new = pd.DataFrame(rows)
    path = OUT / "summary.csv"
    if path.exists():  # keep other runs and visit counts
        old = pd.read_csv(path)
        for c in ("w_orient", "w_relief"):
            old[c] = old[c].fillna(0.0) if c in old else 0.0
        old = old[~(old.run.isin(new.run) & (old.visits == args.visits) & (old.w_orient == args.w_orient) & (old.w_relief == args.w_relief))]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)
    base = pd.read_csv(NEW / "eval" / "summary.csv")
    merged = new.merge(base, on=["run", "split"], suffixes=("", "_base"))
    cols = ["iou_prima", "iou_student", "iou_prima_refined", "iou_student_refined", "iou_fit",
            "lidar_held_prima", "lidar_held_student", "lidar_held_prima_refined", "lidar_held_student_refined", "lidar_held_fit",
            "lidar_held_flat", "distance_rel_error_student", "distance_rel_error_student_refined"]
    merged[["w_orient", "w_relief"]] = merged[["w_orient", "w_relief"]].fillna(0.0)
    summary = merged.groupby(["split", "visits", "w_orient", "w_relief"])[cols].median()
    (OUT / "summary.json").write_text(json.dumps({str(k): v for k, v in summary.to_dict(orient="index").items()}, indent=2))
    print(summary.round(3).T.to_string())


if __name__ == "__main__":
    main()
