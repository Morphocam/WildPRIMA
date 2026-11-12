"""Joint angles and gait for a finished run: python kinematics.py outputs/<run> [outputs/<run> ...]

Writes kinematics.csv (per instance), gait.json (per animal) and kinematics.png into each run folder, and prints a
one-line summary per animal. See animal3d/kinematics.py for what the angles mean and their limits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from animal3d.body import BODY_BOUNDS, AnimalBody
from animal3d.kinematics import JOINTS, LIMBS, gait_summary, joint_angles, posed_joints
from run import ROOT, load_smal

FPS = 10.0  # WildDepth snippets
MIN_IOU, MIN_BOX_H = 0.7, 80  # smaller or poorer fits say little about the legs


def rest_joints(fit, smal) -> np.ndarray:
    body = AnimalBody(**smal, betas=fit["betas"], bounds=BODY_BOUNDS)
    with torch.no_grad():
        body.log_scale.copy_(torch.as_tensor(fit["log_scale"]))
        return body.rest()[1].numpy()


def analyse(out: Path, smal: dict, fps: float) -> list[dict]:
    fit = np.load(out / "fit.npz")
    metrics = pd.read_csv(out / "metrics.csv")
    joints = posed_joints(rest_joints(fit, smal), smal["parents"], fit["pose"])
    table = joint_angles(joints)
    # Time from the frame numbers in the file names when they are numbers, so dropped frames still count as time.
    names = pd.to_numeric(metrics["name"], errors="coerce")
    table.insert(0, "frame", names.astype(int).to_numpy() if names.notna().all() else fit["frame"])
    table.insert(1, "track_id", fit["track_id"])
    table["box_h"], table["iou"] = metrics["box_h"].to_numpy(), metrics["iou"].to_numpy()
    table["usable"] = (~metrics.truncated & ~metrics.overlapping).to_numpy() & (table.iou >= MIN_IOU) & (table.box_h >= MIN_BOX_H)

    # Ground speed: camera-space translation, in metres from the animal's own LiDAR scale when it has one.
    scale = metrics.groupby("track_id")["metres_per_unit"].median() if "metres_per_unit" in metrics else pd.Series(dtype=float)
    table["speed_m_s"] = np.nan
    table["x_m"], table["z_m"], table["shoulder_height_m"] = np.nan, np.nan, np.nan
    summary = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    heights = {t["track_id"]: t.get("shoulder_height_m") for t in summary.get("lidar", {}).get("tracks", [])}
    for track, height in heights.items():
        if height is not None:
            table.loc[table.track_id == track, "shoulder_height_m"] = height
    for track, group in table.groupby("track_id"):
        g = group.sort_values("frame")
        mpu = scale.get(track, np.nan)
        if not np.isfinite(mpu) or len(g) < 2:
            continue
        t = fit["cam_t"][g.index.to_numpy()] * mpu
        table.loc[g.index, "x_m"], table.loc[g.index, "z_m"] = t[:, 0], t[:, 2]
        dt = np.diff(g["frame"].to_numpy()) / fps
        step = np.linalg.norm(np.diff(t[:, [0, 2]], axis=0), axis=1) / dt  # ground plane: x and depth
        table.loc[g.index[1:], "speed_m_s"] = pd.Series(step, index=g.index[1:]).rolling(5, center=True, min_periods=1).median().to_numpy()
    table.to_csv(out / "kinematics.csv", index=False)

    gait = gait_summary(table, fps)
    (out / "gait.json").write_text(json.dumps({"fps": fps, "min_iou": MIN_IOU, "min_box_h": MIN_BOX_H, "tracks": gait}, indent=2))
    if gait:
        plot(table, gait, out)
    return gait


def plot(table: pd.DataFrame, gait: list[dict], out: Path) -> None:
    best = max(gait, key=lambda r: r["frames"])
    g = table[(table.track_id == best["track_id"]) & table.usable]
    g = g[(g.frame >= best["first_frame"]) & (g.frame <= best["last_frame"])].sort_values("frame")
    t = (g.frame - g.frame.iloc[0]) / FPS
    colors = {"left": "#2a6fdb", "right": "#d9822b"}
    fig, axes = plt.subplots(3, 1, figsize=(8, 7.5), sharex=True)
    for ax, (title, names) in zip(axes, [("Limb swing (forward +)", ["left_hind", "right_hind"]),
                                          ("Stifle (knee) inner angle", ["left_stifle", "right_stifle"]),
                                          ("Elbow inner angle", ["left_elbow", "right_elbow"])]):
        for name in names:
            side = name.split("_")[0]
            ax.plot(t, g[name], lw=2, color=colors[side], label=side)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel("degrees")
        ax.grid(alpha=0.25, lw=0.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].legend(frameon=False, ncol=2, loc="upper right")
    axes[-1].set_xlabel("seconds")
    speed = f"{best['speed_m_s']:.2f} m/s" if best["speed_m_s"] is not None else "speed unknown"
    fig.suptitle(f"{out.name}: animal {best['track_id']}, {best['frames']} frames, {speed}, "
                 f"stride {best['stride_hz']:.2f} Hz (periodicity {best['stride_periodicity']:.2f})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "kinematics.png", dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--fps", type=float, default=FPS)
    args = parser.parse_args()
    smal = load_smal(ROOT / "third_party/PRIMA")
    for out in args.runs:
        gait = analyse(out, smal, args.fps)
        if not gait:
            print(f"{out.name}: no animal with {MIN_BOX_H}+ px, IoU {MIN_IOU}+ over 20 consecutive frames")
        for r in gait:
            speed = f"{r['speed_m_s']:.2f}" if r["speed_m_s"] is not None else "  ? "
            print(f"{out.name:20s} animal {r['track_id']:2d}: {r['frames']:4d} frames, {speed} m/s, stride {r['stride_hz']:.2f} Hz "
                  f"(periodicity {r['stride_periodicity']:.2f}, L/R hind corr {r['left_right_hind_corr']:+.2f}), "
                  f"stifle {r['left_stifle_median']:.0f}° ±{r['left_stifle_range'] / 2:.0f}, elbow {r['left_elbow_median']:.0f}° ±{r['left_elbow_range'] / 2:.0f}")


if __name__ == "__main__":
    main()
