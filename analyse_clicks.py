"""Compare hand-clicked 2D keypoints (click_keypoints.py) with the fitted model's projected joints, and with PRIMA's.

    python analyse_clicks.py [--clicks new_output/keypoint_clicks/clicks.json]

For each clicked frame: the fitted (and PRIMA) joints are posed exactly as in the fit, projected with the camera the
fit used, and compared with the clicks on the near side. The near side is the one whose leg joints the fitted model puts
closer to the camera, unless the clicker chose a side. Writes to the clicks folder:
  per_point.csv   pixel error per keypoint, normalised by the box height
  per_angle.csv   image-plane angles (elbow, carpus, stifle, hock, neck, fore/hind swing) from clicks vs fit vs PRIMA
  summary.json    medians, PCK, angle errors and correlations; overlays/ one picture per frame
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from animal3d.body import BODY_BOUNDS, AnimalBody
from animal3d.sources import CALIBRATIONS, Camera
from click_keypoints import run_dir
import pickle

from run import ROOT, full_mask, load_smal, prima_camera

NEW = ROOT / "new_output"
# Clicked point -> (left, right) SMAL joint, after animal3d/kinematics.py (7-10 / 17-20 left, 11-14 / 21-24 right).
SIDE_JOINTS = {"shoulder": (7, 11), "elbow": (8, 12), "carpus": (9, 13), "fore_fetlock": (10, 14),
               "hip": (17, 21), "stifle": (18, 22), "hock": (19, 23), "hind_fetlock": (20, 24),
               "withers": (15, 15), "poll": (16, 16)}
ANGLES = {"elbow": ("shoulder", "elbow", "carpus"), "carpus": ("elbow", "carpus", "fore_fetlock"),
          "stifle": ("hip", "stifle", "hock"), "hock": ("stifle", "hock", "hind_fetlock")}


def fit_camera(run: str, width: int, height: int) -> Camera:
    calib = json.loads((run_dir(run) / "summary.json").read_text())["input"]["calibration"]
    if calib["name"] in CALIBRATIONS:
        c = CALIBRATIONS[calib["name"]]
        sx, sy = width / c.image_size[0], height / c.image_size[1]
        return Camera(c.camera.fx * sx, c.camera.fy * sy, c.camera.cx * sx, c.camera.cy * sy)
    return Camera(calib["fx"], calib["fy"], calib["cx"], calib["cy"])  # a calibration.json made for these frames


def project(points: np.ndarray, cam: Camera) -> np.ndarray:
    return np.stack([cam.fx * points[:, 0] / points[:, 2] + cam.cx, cam.fy * points[:, 1] / points[:, 2] + cam.cy], 1)


def posed(smal, betas, log_scale, orient, pose) -> np.ndarray:
    body = AnimalBody(**smal, betas=betas, bounds=BODY_BOUNDS)
    with torch.no_grad():
        if log_scale is not None:
            body.log_scale.copy_(torch.as_tensor(log_scale))
        j = body.posed_joints(torch.as_tensor(orient[None], dtype=torch.float32), torch.as_tensor(pose[None], dtype=torch.float32))
    return j[0].numpy()


def angle(a, b, c) -> float:
    u, v = np.asarray(a) - b, np.asarray(c) - b
    return float(np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1))))


def neck_angle(withers, poll) -> float:
    """Image-plane neck elevation above horizontal, independent of facing direction (degrees)."""
    d = np.asarray(poll) - withers
    return float(np.degrees(np.arctan2(-d[1], abs(d[0]))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clicks", type=Path, default=NEW / "keypoint_clicks" / "clicks.json")
    args = parser.parse_args()
    out = args.clicks.parent
    (out / "overlays").mkdir(exist_ok=True)
    doc = json.loads(args.clicks.read_text())
    smal = load_smal(ROOT / "third_party/PRIMA")
    cache: dict[str, tuple] = {}
    point_rows, angle_rows, frames_used = [], [], []
    for fid, entry in doc["frames"].items():
        rec = entry["record"]
        if rec.get("unusable") or not rec["points"]:
            continue
        run = entry["run"]
        if run not in cache:
            d = run_dir(run)
            cache[run] = (np.load(d / "fit.npz"), np.load(d / "baseline.npz"), pd.read_csv(d / "metrics.csv"),
                          pickle.loads((d / "masks.pkl").read_bytes()))
        fit, base, metrics, masks = cache[run]
        hit = np.where((pd.to_numeric(metrics["name"]) == entry["frame"]) & (metrics.track_id == entry["track_id"]))[0]
        if len(hit) != 1:
            print(f"{fid}: no unique fitted instance, skipped")
            continue
        i = int(hit[0])
        w, h = int(metrics.at[i, "width"]), int(metrics.at[i, "height"])
        box_h = float(metrics.at[i, "box_h"])
        cam = fit_camera(run, w, h)
        j_fit = posed(smal, fit["betas"], fit["log_scale"], fit["global_orient"][i], fit["pose"][i]) + fit["cam_t"][i]
        j_prima = posed(smal, base["betas"][i], None, base["global_orient"][i], base["pose"][i]) + base["cam_t"][i]
        # The fitted joints must sit on the fitted mesh: check against the projected vertices' box.
        uv_v = project(fit["vertices"][i] + fit["cam_t"][i], cam)
        uv_fit, uv_prima = project(j_fit, cam), project(j_prima, prima_camera(w, h))
        legs_l, legs_r = [7, 8, 9, 10, 17, 18, 19, 20], [11, 12, 13, 14, 21, 22, 23, 24]
        near = rec.get("side") or ("left" if j_fit[legs_l, 2].mean() < j_fit[legs_r, 2].mean() else "right")
        s = 0 if near == "left" else 1
        clicks = {k: (np.asarray(v, float) if v is not None else None) for k, v in rec["points"].items()}
        # Which side's projected joints lie closer to the clicks (a check on the depth-based choice).
        dist = [np.nanmedian([np.linalg.norm(clicks[k] - uv_fit[SIDE_JOINTS[k][side]]) for k in SIDE_JOINTS
                              if clicks.get(k) is not None and SIDE_JOINTS[k][0] != SIDE_JOINTS[k][1]]) for side in (0, 1)]
        mask = full_mask(masks[i], h, w)
        on_mask = lambda uv: float(np.mean([mask[int(np.clip(y, 0, h - 1)), int(np.clip(x, 0, w - 1))] for x, y in uv]))
        clicked_uv = np.array([v for v in clicks.values() if v is not None])
        frames_used.append({"id": fid, "fit_leg_joints_on_mask": on_mask(uv_fit[legs_l + legs_r]),
                            "prima_leg_joints_on_mask": on_mask(uv_prima[legs_l + legs_r]), "clicks_on_mask": on_mask(clicked_uv), "species": entry["species"], "near_side": near, "closer_side_2d": "left" if dist[0] <= dist[1] else "right",
                            "box_h": box_h, "fit_box_h": float(uv_v[:, 1].max() - uv_v[:, 1].min()),
                            "iou": float(metrics.at[i, "iou"]), "prima_iou": float(metrics.at[i, "prima_iou"])})
        for key, (jl, jr) in SIDE_JOINTS.items():
            c = clicks.get(key)
            if c is None:
                continue
            j = (jl, jr)[s]
            point_rows.append({"id": fid, "species": entry["species"], "point": key,
                               "err_fit": float(np.linalg.norm(c - uv_fit[j])) / box_h,
                               "err_prima": float(np.linalg.norm(c - uv_prima[j])) / box_h})
        get = lambda uv, k: uv[SIDE_JOINTS[k][s]]
        for name, (a, b, c) in ANGLES.items():
            if any(clicks.get(k) is None for k in (a, b, c)):
                continue
            angle_rows.append({"id": fid, "species": entry["species"], "angle": name,
                               "clicked": angle(clicks[a], clicks[b], clicks[c]),
                               "fit": angle(get(uv_fit, a), get(uv_fit, b), get(uv_fit, c)),
                               "prima": angle(get(uv_prima, a), get(uv_prima, b), get(uv_prima, c))})
        # Limb swing in the image: top joint -> fetlock against image vertical, positive = foot forward. Forward is the
        # clicked hip -> shoulder direction, so the sign does not depend on the fit.
        if clicks.get("shoulder") is not None and clicks.get("hip") is not None:
            fwd = np.sign(clicks["shoulder"][0] - clicks["hip"][0]) or 1.0
            for name, (top, foot) in {"fore_swing": ("shoulder", "fore_fetlock"), "hind_swing": ("hip", "hind_fetlock")}.items():
                if clicks.get(top) is None or clicks.get(foot) is None:
                    continue
                sw = lambda t, f: float(np.degrees(np.arctan2(fwd * (f[0] - t[0]), f[1] - t[1])))
                angle_rows.append({"id": fid, "species": entry["species"], "angle": name, "clicked": sw(clicks[top], clicks[foot]),
                                   "fit": sw(get(uv_fit, top), get(uv_fit, foot)), "prima": sw(get(uv_prima, top), get(uv_prima, foot))})
        if clicks.get("withers") is not None and clicks.get("poll") is not None:
            angle_rows.append({"id": fid, "species": entry["species"], "angle": "neck",
                               "clicked": neck_angle(clicks["withers"], clicks["poll"]),
                               "fit": neck_angle(uv_fit[15], uv_fit[16]), "prima": neck_angle(uv_prima[15], uv_prima[16])})
        # Overlay: clicks (green), fitted joints (orange), PRIMA (blue), on the near side.
        img = cv2.imread(entry["image"])
        x1, y1, x2, y2 = entry["box_xyxy"]
        pad = 0.3 * max(x2 - x1, y2 - y1)
        X0, Y0 = int(max(0, x1 - pad)), int(max(0, y1 - pad))
        crop = img[Y0:int(min(h, y2 + pad)), X0:int(min(w, x2 + pad))].copy()
        zoom = 600 / max(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=zoom, fy=zoom)
        P = lambda p: (int((p[0] - X0) * zoom), int((p[1] - Y0) * zoom))
        chains = [["shoulder", "elbow", "carpus", "fore_fetlock"], ["hip", "stifle", "hock", "hind_fetlock"], ["withers", "poll"]]
        for colour, pts in (((60, 200, 60), {k: v for k, v in clicks.items() if v is not None}),
                            ((0, 140, 255), {k: get(uv_fit, k) for k in SIDE_JOINTS}),
                            ((255, 120, 40), {k: get(uv_prima, k) for k in SIDE_JOINTS})):
            for chain in chains:
                ok = [k for k in chain if k in pts]
                for a, b in zip(ok[:-1], ok[1:]):
                    cv2.line(crop, P(pts[a]), P(pts[b]), colour, 2, cv2.LINE_AA)
                for k in ok:
                    cv2.circle(crop, P(pts[k]), 4, colour, -1, cv2.LINE_AA)
        cv2.putText(crop, f"{entry['species']} {near} side  green=clicked orange=ours blue=PRIMA", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out / "overlays" / (fid.replace("/", "_") + ".jpg")), crop)

    pts, ang, frm = pd.DataFrame(point_rows), pd.DataFrame(angle_rows), pd.DataFrame(frames_used)
    pts.to_csv(out / "per_point.csv", index=False)
    ang.to_csv(out / "per_angle.csv", index=False)
    frm.to_csv(out / "frames.csv", index=False)
    ang["abs_err_fit"], ang["abs_err_prima"] = (ang.fit - ang.clicked).abs(), (ang.prima - ang.clicked).abs()
    summary = {
        "frames": len(frm), "frames_by_species": frm.species.value_counts().to_dict(),
        "joints_on_mask_mean": {k: float(frm[f"{k}_on_mask"].mean()) for k in ("fit_leg_joints", "prima_leg_joints", "clicks")},
        "near_side_matches_closer_2d": float((frm.near_side == frm.closer_side_2d).mean()),
        "fit_box_h_over_det_box_h_median": float((frm.fit_box_h / frm.box_h).median()),
        "point_err_over_box_h_median": {"fit": float(pts.err_fit.median()), "prima": float(pts.err_prima.median())},
        "pck@0.1": {"fit": float((pts.err_fit < 0.1).mean()), "prima": float((pts.err_prima < 0.1).mean())},
        "pck@0.05": {"fit": float((pts.err_fit < 0.05).mean()), "prima": float((pts.err_prima < 0.05).mean())},
        "per_point_median": pts.groupby("point")[["err_fit", "err_prima"]].median().round(3).to_dict("index"),
        "per_angle": {},
    }
    for name, g in ang.groupby("angle"):
        summary["per_angle"][name] = {
            "n": len(g), "clicked_median": float(g.clicked.median()), "fit_median": float(g.fit.median()),
            "mae_fit": float(g.abs_err_fit.median()), "mae_prima": float(g.abs_err_prima.median()),
            "bias_fit": float((g.fit - g.clicked).median()),
            "mae_fit_debiased": float(((g.fit - g.clicked) - (g.fit - g.clicked).median()).abs().median()),
            "mae_prima_debiased": float(((g.prima - g.clicked) - (g.prima - g.clicked).median()).abs().median()),
            "corr_fit": float(np.corrcoef(g.clicked, g.fit)[0, 1]) if len(g) > 2 else None,
            "corr_prima": float(np.corrcoef(g.clicked, g.prima)[0, 1]) if len(g) > 2 else None,
        }
    legs = ang[ang.angle.isin(list(ANGLES))]
    summary["leg_angles_all"] = {"n": len(legs), "mae_fit": float(legs.abs_err_fit.median()), "mae_prima": float(legs.abs_err_prima.median()),
                                 "corr_fit": float(np.corrcoef(legs.clicked, legs.fit)[0, 1]),
                                 "corr_prima": float(np.corrcoef(legs.clicked, legs.prima)[0, 1])}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
