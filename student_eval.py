"""Step 4c: score the student on held-out runs exactly as the fits are scored, beside PRIMA and the fit.

Called as `python student.py eval [RUN ...]` (default: every val and test run in the cache). Writes
new_output/eval/<run>/metrics.csv and new_output/eval/summary.json. The student's mesh has no horns or humps (they come
from the species table at inference), so its IoU on rhinos and camels is a little handicapped against the fit's.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch

from pose_prior import NEW, ROOT, run_folders, split_of
from student import CACHE, BatchedBody, Data, Student, camera_translation, load_prima


@torch.no_grad()
def evaluate(names: list[str], device: str) -> None:
    import run as R
    from animal3d.sources import open_source

    cached = sorted(p.name for p in CACHE.iterdir() if (p / "meta.npz").exists())
    names = names or [n for n in cached if split_of(n) in ("val", "test")]
    prima = load_prima(device)
    student = Student(prima.model.smal_head).to(device)
    faces = prima.faces.astype(np.int64)
    del prima
    ckpt = torch.load(NEW / "student" / "student.pt", map_location=device)
    student.load_state_dict(ckpt["state"])
    student.eval()
    body = BatchedBody(R.load_smal(ROOT / "third_party/PRIMA")).to(device)
    rows = []
    for name in names:
        out, inp, calib = run_folders()[name]
        data = Data([name], device, usable_only=False)
        m = data.meta
        verts, trans, mpu, seconds = [], [], [], 0.0
        for s in range(0, data.n, 32):
            idx = torch.arange(s, min(s + 32, data.n), device=device)
            torch.cuda.synchronize()
            t0 = time.time()
            o = student(data.tokens[idx.cpu()].to(device).float())
            t = camera_translation(o["cam"], m, idx)
            v = body(o["betas"], o["log_scale"], o["orient"], o["pose"])
            torch.cuda.synchronize()
            seconds += time.time() - t0
            verts.append(v.cpu().numpy())
            trans.append(t.cpu().numpy())
            mpu.append(o["log_mpu"].exp().cpu().numpy())
        verts, trans, mpu = np.concatenate(verts), np.concatenate(trans), np.concatenate(mpu)
        source = open_source(inp, None, calib)
        df, _, masks, _ = R.load_prima_results(out)
        scored = R.score_meshes(source, df, masks, faces, verts, trans, lambda w, h: R.fit_camera(source, w, h))
        metrics = pd.read_csv(out / "metrics.csv")
        scored["student_mpu"] = mpu
        scored["student_distance_m"] = trans[:, 2] * mpu
        scored["lidar_depth_m"] = metrics["lidar_depth_m"].to_numpy()
        target = NEW / "eval" / name
        target.mkdir(parents=True, exist_ok=True)
        scored.to_csv(target / "metrics.csv", index=False)
        clean = (~metrics.truncated & ~metrics.overlapping).to_numpy()
        med = lambda frame, col: float(np.nanmedian(frame.loc[clean, col])) if col in frame and clean.any() else None
        depth = scored["lidar_depth_m"].to_numpy()
        ok = clean & np.isfinite(depth) & (depth > 0)
        fit_mpu = metrics["metres_per_unit"].groupby(metrics.track_id).transform("median").to_numpy() if "metres_per_unit" in metrics else np.full(len(df), np.nan)
        rows.append({
            "run": name, "split": split_of(name), "instances": int(len(df)), "clean": int(clean.sum()),
            "iou_prima": med(metrics, "prima_iou"), "iou_fit": med(metrics, "iou"), "iou_student": med(scored, "iou"),
            "lidar_held_prima": med(metrics, "prima_relief_error_held_m"), "lidar_held_fit": med(metrics, "relief_error_held_m"),
            "lidar_held_student": med(scored, "relief_error_held_m"), "lidar_held_flat": med(metrics, "flat_relief_held_m"),
            "distance_rel_error_student": float(np.nanmedian(np.abs(scored["student_distance_m"][ok] - depth[ok]) / depth[ok])) if ok.any() else None,
            "mpu_rel_error_student_vs_fit": float(np.nanmedian(np.abs(mpu[clean] - fit_mpu[clean]) / fit_mpu[clean])) if np.isfinite(fit_mpu[clean]).any() else None,
            "student_ms_per_instance": 1000 * seconds / max(len(df), 1),
        })
        print(json.dumps(rows[-1]))
    summary = pd.DataFrame(rows)
    (NEW / "eval").mkdir(parents=True, exist_ok=True)
    summary.to_csv(NEW / "eval" / "summary.csv", index=False)
    by_split = summary.groupby("split").median(numeric_only=True)
    (NEW / "eval" / "summary.json").write_text(json.dumps({"runs": rows, "median_by_split": by_split.to_dict(orient="index"),
                                                           "checkpoint_step": ckpt["step"]}, indent=2))
    print(by_split.round(3).T)
