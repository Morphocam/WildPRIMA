"""Item 7: the released AniMer (Lyu et al., CVPR 2025) as a second single-image baseline, scored exactly like PRIMA.

    python baseline_animer.py [--set table1|heldout|all] [RUN ...]

Same detections, boxes and masks as PRIMA and the fits (each run's detections.csv / masks.pkl), AniMer's own crop and
camera convention (focal = 1000 / 256 x the longer image side, principal point at the centre, as in its demo), and
run.score_meshes for silhouette IoU and the held-out LiDAR relief error. Like PRIMA, AniMer has no metric scale.
Writes new_output/baseline_animer/<run>/metrics.csv and new_output/baseline_animer/summary.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from pose_prior import EXISTING, NEW, ROOT, run_folders, split_of

sys.path.insert(0, str(ROOT / "third_party"))
OUT = NEW / "baseline_animer"


@torch.no_grad()
def predict_run(name, model, cfg, device):
    import run as R
    from amr.datasets.vitdet_dataset import ViTDetDataset
    from amr.utils import recursive_to
    from amr.utils.renderer import cam_crop_to_full
    from animal3d.sources import Camera, open_source

    out, inp, calib = run_folders()[name]
    df, _, masks, _ = R.load_prima_results(out)
    source = open_source(inp, None, calib)
    n = len(df)
    verts = np.zeros((n, 3889, 3), np.float32)
    trans = np.zeros((n, 3), np.float32)
    by_frame = {f: g.index.to_numpy() for f, g in df.groupby("frame")}
    for frame in source:
        rows = by_frame.get(frame.index)
        if rows is None:
            continue
        rgb = cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2RGB)  # AniMer's demo feeds RGB
        boxes = df.loc[rows, ["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
        loader = torch.utils.data.DataLoader(ViTDetDataset(cfg, rgb, boxes), batch_size=8, shuffle=False, num_workers=0)
        k = 0
        for batch in loader:
            batch = recursive_to(batch, device)
            o = model(batch)
            img_size = batch["img_size"].float()
            focal = cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * img_size.max()
            t = cam_crop_to_full(o["pred_cam"], batch["box_center"].float(), batch["box_size"].float(), img_size, focal)
            b = len(t)
            verts[rows[k:k + b]] = o["pred_vertices"].float().cpu().numpy()
            trans[rows[k:k + b]] = t.float().cpu().numpy()
            k += b

    def camera_for(w, h):
        f = cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * max(w, h)
        return Camera(f, f, w / 2, h / 2)

    scored = R.score_meshes(source, df, masks, np.asarray(torch.as_tensor(model.smal.faces).cpu().numpy(), dtype=np.int64), verts, trans, camera_for)
    target = OUT / name
    target.mkdir(parents=True, exist_ok=True)
    scored.to_csv(target / "metrics.csv", index=False)
    metrics = pd.read_csv(out / "metrics.csv")
    clean = (~metrics.truncated & ~metrics.overlapping).to_numpy()
    med = lambda frame, col: float(np.nanmedian(frame.loc[clean, col])) if col in frame and clean.any() else None
    return {"run": name, "set": "table1" if name in EXISTING else split_of(name), "clean": int(clean.sum()),
            "iou_animer": med(scored, "iou"), "iou_prima": med(metrics, "prima_iou"), "iou_fit": med(metrics, "iou"),
            "lidar_held_animer": med(scored, "relief_error_held_m"), "lidar_held_prima": med(metrics, "prima_relief_error_held_m"),
            "lidar_held_fit": med(metrics, "relief_error_held_m"), "lidar_held_flat": med(metrics, "flat_relief_held_m")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*")
    parser.add_argument("--set", default="all", choices=["table1", "heldout", "all"])
    args = parser.parse_args()
    from animer_loader import load_animer

    device = "cuda"
    model, cfg = load_animer(device)
    names = args.runs or [n for n in run_folders()
                          if (args.set in ("table1", "all") and n in EXISTING)
                          or (args.set in ("heldout", "all") and split_of(n) in ("val", "test"))]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        rows.append(predict_run(name, model, cfg, device))
        print(rows[-1], flush=True)
    new = pd.DataFrame(rows)
    path = OUT / "summary.csv"
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[~old.run.isin(new.run)], new], ignore_index=True)
    new.to_csv(path, index=False)
    print(new.groupby("set").median(numeric_only=True).round(3).T.to_string())


if __name__ == "__main__":
    main()
