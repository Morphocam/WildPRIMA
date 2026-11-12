"""Paper figures rendered from the saved fits.

    python make_figures.py teaser      # -> paper/fig/teaser.jpg (+ panels in new_output/figures/teaser/)
    python make_figures.py pipeline    # -> paper/fig/pipeline.pdf (+ thumbnails in new_output/figures/pipeline/)

For each clip, the instance shown is chosen automatically: clean (not cut off, not overlapping), large, and with a
high fitted IoU; ties go to the largest gain over PRIMA. The choice is written to new_output/figures/teaser/choice.json.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import pandas as pd
import pyrender
import trimesh

import run as R
from animal3d.render import FrameRenderer
from animal3d.sources import open_source
from pose_prior import NEW, ROOT, run_folders

OUT = NEW / "figures"
TEASER = [("giraffe_4_lidar", "Giraffe"), ("Rhino_3_lidar", "White rhinoceros"),
          ("Camel_1_domes", "Bactrian camel"), ("Ankle_cow_5_lidar", "Ankole-Watusi cattle")]
# Rows chosen by eye from contact sheets, where the automatic choice was occluded or front-on (all camels face the camera).
PINNED = {"Rhino_3_lidar": 105, "Camel_1_domes": 394}
PRIMA_COLOR = (0.55, 0.64, 0.80)
OURS_COLOR = (0.93, 0.62, 0.33)
TILE = 360


def pick_instance(metrics: pd.DataFrame, df: pd.DataFrame) -> int:
    h = (df.y2 - df.y1).to_numpy()
    clean = (~metrics.truncated & ~metrics.overlapping).to_numpy()
    ok = clean & (h >= 0.75 * h[clean].max())  # the largest animals only: the figure must show detail
    side = clean & (h >= 0.5 * h[clean].max()) & ((df.x2 - df.x1).to_numpy() >= 1.15 * h)  # side-on views, where the 3D pose is not a depth guess
    ok = side if side.any() else ok
    score = metrics.iou.to_numpy() + 0.5 * (metrics.iou - metrics.prima_iou).to_numpy()
    score[~ok] = -np.inf
    return int(np.argmax(score))


def frame_rgb(source, index: int) -> np.ndarray:
    for frame in source:
        if frame.index == index:
            return cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2RGB)
    raise KeyError(index)


def square(image: np.ndarray, box: np.ndarray, pad: float = 1.2) -> np.ndarray:
    """Square crop around the box, padded with the image's own edge pixels, resized to TILE."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    half = max(box[2] - box[0], box[3] - box[1]) * pad / 2
    x0, y0, x1, y1 = (int(round(v)) for v in (cx - half, cy - half, cx + half, cy + half))
    padded = cv2.copyMakeBorder(image, max(0, -y0), max(0, y1 - image.shape[0]), max(0, -x0), max(0, x1 - image.shape[1]),
                                cv2.BORDER_REPLICATE)
    crop = padded[y0 + max(0, -y0): y1 + max(0, -y0), x0 + max(0, -x0): x1 + max(0, -x0)]
    return cv2.resize(crop, (TILE, TILE), interpolation=cv2.INTER_AREA)


_three_quarter = None
_alive: list = []


_REGRESSOR = None


def level(body: np.ndarray) -> np.ndarray:
    """Rotate so hips->shoulders is +x and the left-right axis is horizontal; centred. Grazing poses tilt the root."""
    global _REGRESSOR
    if _REGRESSOR is None:
        _REGRESSOR = R.load_smal(ROOT / "third_party/PRIMA")["j_regressor"]
    j = _REGRESSOR @ body[: _REGRESSOR.shape[1]]
    shoulders, hips = (j[7] + j[11]) / 2, (j[17] + j[21]) / 2
    left = (j[7] + j[17]) / 2 - (j[11] + j[21]) / 2
    x = shoulders - hips
    x[2] *= 0.3  # keep some natural pitch, remove most of it
    x /= np.linalg.norm(x)
    z = np.cross(x, left)
    z = z if z[2] > 0 else -z
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    out = body @ np.stack([x, y, z]).T
    return out - (out.min(0) + out.max(0)) / 2


def three_quarter(vertices: np.ndarray, faces: np.ndarray, orient: np.ndarray, color, yaw_deg=-35.0, pitch_deg=12.0) -> np.ndarray:
    """The body in its own frame (x nose, z up), seen from the front-left and slightly above, on white."""
    global _three_quarter
    if _three_quarter is None:
        _three_quarter = pyrender.OffscreenRenderer(TILE, TILE)
    body = level(vertices @ orient)  # remove global rotation (row-vector R^T v), then level on shoulders and hips
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    turned = body @ rz.T
    view = np.c_[turned[:, 0], turned[:, 2], -turned[:, 1]]  # x right, y up, z toward viewer: side view
    rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    view = view @ rx.T
    half = float(np.abs(view[:, :2]).max()) * 1.08
    view = view - np.array([0.0, 0.0, 10.0])
    mesh = trimesh.Trimesh(view, faces, process=False)
    material = pyrender.MetallicRoughnessMaterial(baseColorFactor=(*color, 1.0), metallicFactor=0.0, roughnessFactor=0.7)
    scene = pyrender.Scene(ambient_light=(0.45, 0.45, 0.45), bg_color=(1.0, 1.0, 1.0, 1.0))
    scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True))
    scene.add(pyrender.OrthographicCamera(xmag=half, ymag=half, znear=0.01, zfar=100.0))
    for direction, power in (((0.3, 0.8, 1.0), 2.5), ((-0.8, 0.3, 0.6), 1.2)):
        d = np.array(direction) / np.linalg.norm(direction)
        z = d
        x = np.cross([0, 1, 0], z)
        x = x / np.linalg.norm(x)
        pose = np.eye(4)
        pose[:3, :3] = np.c_[x, np.cross(z, x), z]
        scene.add(pyrender.DirectionalLight(intensity=power), pose=pose)
    color_img, _ = _three_quarter.render(scene)
    return color_img[..., :3].copy()


def canonical(out, pose: np.ndarray) -> np.ndarray:
    """The fitted body re-posed with identity global rotation, in SMAL's axes (x nose, z up)."""
    import torch
    from pose_prior import rebuild_body
    body = rebuild_body(out, R.load_smal(ROOT / "third_party/PRIMA"), "cpu")
    with torch.no_grad():
        return body(torch.eye(3)[None], torch.as_tensor(pose)[None].float())[0].numpy()


def put(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (10, TILE - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, (10, TILE - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def teaser() -> None:
    target = OUT / "teaser"
    target.mkdir(parents=True, exist_ok=True)
    runs = run_folders()
    rows, choice = [], {}
    for name, title in TEASER:
        out, inp, calib = runs[name]
        source = open_source(inp, None, calib)
        df, _, _, _ = R.load_prima_results(out)
        metrics = pd.read_csv(out / "metrics.csv")
        fit, base = np.load(out / "fit.npz"), np.load(out / "baseline.npz")
        i = PINNED.get(name, pick_instance(metrics, df))
        rgb = frame_rgb(source, int(df.frame.iloc[i]))
        h, w = rgb.shape[:2]
        box = df.loc[i, ["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
        keep = [FrameRenderer(w, h, R.prima_camera(w, h), base["faces"]), FrameRenderer(w, h, R.fit_camera(source, w, h), fit["faces"])]
        _alive.extend(keep)  # never delete an EGL renderer (see FrameRenderer)
        prima = keep[0].overlay(rgb, base["vertices"][i:i + 1], base["cam_t"][i:i + 1], [PRIMA_COLOR], alpha=0.75)
        ours = keep[1].overlay(rgb, fit["vertices"][i:i + 1], fit["cam_t"][i:i + 1], [OURS_COLOR], alpha=0.75)
        tiles = [square(rgb, box), put(square(prima, box), f"IoU {metrics.prima_iou[i]:.2f}"), put(square(ours, box), f"IoU {metrics.iou[i]:.2f}"),
                 three_quarter(base["vertices"][i], base["faces"], base["global_orient"][i], PRIMA_COLOR),
                 three_quarter(canonical(out, fit["pose"][i]), fit["faces"], np.eye(3), OURS_COLOR)]
        for k, tile in zip(("input", "prima", "ours", "prima3d", "ours3d"), tiles):
            cv2.imwrite(str(target / f"{name}_{k}.png"), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
        gap = np.full((TILE, 6, 3), 255, np.uint8)
        rows.append(np.concatenate(sum(([t, gap] for t in tiles), [])[:-1], axis=1))
        choice[name] = {"title": title, "row": i, "frame": int(df.frame.iloc[i]), "track_id": int(df.track_id.iloc[i]),
                        "iou": float(metrics.iou[i]), "prima_iou": float(metrics.prima_iou[i])}
        print(name, choice[name])
    # Clips side by side in two columns of rows, so the strip stays wide and short.
    gap_v = np.full((6, rows[0].shape[1], 3), 255, np.uint8)
    left = np.concatenate([rows[0], gap_v, rows[2]], axis=0)
    right = np.concatenate([rows[1], gap_v, rows[3]], axis=0)
    grid = np.concatenate([left, np.full((left.shape[0], 24, 3), 255, np.uint8), right], axis=1)
    cv2.imwrite(str(target / "teaser.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    small = cv2.resize(grid, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(ROOT / "paper" / "fig" / "teaser.jpg"), cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    (target / "choice.json").write_text(json.dumps(choice, indent=2))


def pipeline_thumbnails(name: str = "giraffe_4_lidar") -> dict[str, np.ndarray]:
    """Images for the pipeline figure, all from the teaser instance of one clip."""
    target = OUT / "pipeline"
    target.mkdir(parents=True, exist_ok=True)
    out, inp, calib = run_folders()[name]
    source = open_source(inp, None, calib)
    df, base, masks, _ = R.load_prima_results(out)
    metrics = pd.read_csv(out / "metrics.csv")
    fit = np.load(out / "fit.npz")
    i = PINNED.get(name, pick_instance(metrics, df))
    frame = next(f for f in source if f.index == int(df.frame.iloc[i]))
    rgb = cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    box = df.loc[i, ["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
    thumbs = {}
    # Whole frame with every detected animal boxed.
    full = rgb.copy()
    for _, r in df[df.frame == df.frame.iloc[i]].iterrows():
        cv2.rectangle(full, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), (255, 210, 60), 4)
    thumbs["input"] = cv2.resize(full, (TILE * w // h, TILE), interpolation=cv2.INTER_AREA)
    # LiDAR points coloured by depth, dilated so they show at print size.
    lidar = rgb.copy() // 2 + 64
    if frame.depth_m is not None:
        d = frame.depth_m
        ys, xs = np.nonzero(d > 0)
        near, far = np.percentile(d[ys, xs], [5, 95])
        colors = cv2.applyColorMap(np.clip(255 * (d[ys, xs] - near) / (far - near + 1e-6), 0, 255).astype(np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0, ::-1]
        for x, y, c in zip(xs, ys, colors):
            cv2.circle(lidar, (int(x), int(y)), 4, tuple(int(v) for v in c), -1)
    thumbs["lidar"] = square(lidar, box)
    mask = R.full_mask(masks[i], h, w)
    tinted = rgb.copy()
    tinted[mask] = (0.45 * tinted[mask] + 0.55 * np.array([255, 210, 60])).astype(np.uint8)
    thumbs["mask"] = square(tinted, box)
    keep = [FrameRenderer(w, h, R.prima_camera(w, h), base["faces"]), FrameRenderer(w, h, R.fit_camera(source, w, h), fit["faces"])]
    _alive.extend(keep)
    thumbs["prima"] = square(keep[0].overlay(rgb, base["vertices"][i:i + 1], base["cam_t"][i:i + 1], [PRIMA_COLOR], alpha=0.75), box)
    thumbs["ours"] = square(keep[1].overlay(rgb, fit["vertices"][i:i + 1], fit["cam_t"][i:i + 1], [OURS_COLOR], alpha=0.75), box)
    thumbs["ours3d"] = three_quarter(canonical(out, fit["pose"][i]), fit["faces"], np.eye(3), OURS_COLOR)
    thumbs["prima3d"] = three_quarter(base["vertices"][i], base["faces"], base["global_orient"][i], PRIMA_COLOR)
    for k, v in thumbs.items():
        cv2.imwrite(str(target / f"{k}.png"), cv2.cvtColor(v, cv2.COLOR_RGB2BGR))
    return thumbs


def pipeline() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    t = pipeline_thumbnails()
    kin = cv2.cvtColor(cv2.imread(str(NEW / "kinematics_existing" / "giraffe_4_lidar" / "kinematics.png")), cv2.COLOR_BGR2RGB)
    plt.rcParams.update({"font.family": "serif", "font.size": 6.5, "mathtext.fontset": "cm"})
    W, H = 7.0, 2.75
    fig = plt.figure(figsize=(W, H))
    ink, muted = "#222222", "#555555"
    fit_bg, learn_bg = "#fdf3e7", "#eaf1fb"

    def image(img, x, y, w, h):  # figure-fraction box, aspect kept by cropping to it
        ax = fig.add_axes([x, y, w, h])
        target_ratio = (w * W) / (h * H)
        ih, iw = img.shape[:2]
        if iw / ih > target_ratio:
            cut = int((iw - ih * target_ratio) / 2)
            img = img[:, cut: iw - cut]
        else:
            cut = int((ih - iw / target_ratio) / 2)
            img = img[cut: ih - cut]
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.4); sp.set_color("#999999")
        return ax

    def panel(x, y, w, h, color):
        fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012", transform=fig.transFigure,
                                          facecolor=color, edgecolor="none", zorder=-5))

    def arrow(x0, y0, x1, y1, rad=0.0):
        fig.patches.append(FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=6,
                                           linewidth=0.7, color=ink, connectionstyle=f"arc3,rad={rad}"))

    def text(x, y, s, size=6.5, weight="normal", color=ink, ha="center", va="top"):
        fig.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, linespacing=1.15)

    def box(x, y, w, h, title, body, color="white"):
        fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.01", transform=fig.transFigure,
                                          facecolor=color, edgecolor="#888888", linewidth=0.5))
        text(x + w / 2, y + h - 0.025, title, weight="bold")
        text(x + w / 2, y + h - 0.085, body, size=5.6, color=muted)

    # ---- top: per-clip fitting
    panel(0.005, 0.40, 0.99, 0.595, fit_bg)
    text(0.012, 0.985, "Per-clip fitting (Sec. 3.1-3.4)", size=7, weight="bold", ha="left")
    top, ih = 0.545, 0.35
    sq = ih * H / W
    cap = 5.9
    image(t["input"], 0.012, top, 0.13, ih)
    text(0.077, top - 0.012, "RGB video + LiDAR\n(WildDepth, 10 Hz)", size=cap)
    x = 0.165
    arrow(0.145, top + ih / 2, x - 0.003, top + ih / 2)
    image(t["lidar"], x, top + ih / 2 + 0.005, sq / 2 - 0.003, ih / 2 - 0.005)
    image(t["mask"], x, top, sq / 2 - 0.003, ih / 2 - 0.005)
    image(t["prima"], x + sq / 2, top, sq / 2, ih)
    text(x + sq / 2, top - 0.012, "masks, tracks, LiDAR;\nPRIMA pose + camera", size=cap)
    x2 = x + sq + 0.023
    arrow(x + sq + 0.003, top + ih / 2, x2 - 0.003, top + ih / 2)
    bw2 = 0.115
    box(x2, top, bw2, ih, "Species prior", "BioCLIP or label\n$\\rightarrow$ 29-species table\n\nwhich parts exist:\nhorns (paired, nasal),\nhumps, ear size")
    x3 = x2 + bw2 + 0.023
    arrow(x2 + bw2 + 0.003, top + ih / 2, x3 - 0.003, top + ih / 2)
    image(t["prima3d"], x3, top + ih / 2 + 0.005, sq / 2 - 0.003, ih / 2 - 0.005)
    image(t["ours3d"], x3, top, sq / 2 - 0.003, ih / 2 - 0.005)
    text(x3 + sq / 2 + 0.005, top + ih - 0.005, "SMAL $\\rightarrow$ ours\n\nplan search (12)\nbone scales $l_g, t_g$\nsym. offsets\nhorns, humps", size=5.6, color=muted, ha="left")
    text(x3 + sq / 2 + 0.035, top - 0.012, "one body per clip", size=cap)
    x4 = x3 + sq / 2 + 0.105
    arrow(x4 - 0.012, top + ih / 2, x4 - 0.003, top + ih / 2)
    image(t["ours"], x4, top, sq, ih)
    text(x4 + sq / 2, top - 0.012, "pose per instance:\nmask + LiDAR relief", size=cap)
    x5 = x4 + sq + 0.022
    arrow(x4 + sq + 0.003, top + ih / 2, x5 - 0.003, top + ih / 2)
    kin = pd.read_csv(NEW / "kinematics_existing" / "giraffe_4_lidar" / "kinematics.csv")
    track = kin[kin.track_id == 3].sort_values("frame")
    sec = (track.frame - track.frame.iloc[0]).to_numpy() / 10.0
    win = sec <= 6
    ax = fig.add_axes([x5 + 0.04, top + 0.035, 0.985 - x5 - 0.04, ih - 0.035])
    ax.plot(sec[win], track.left_stifle[win], color="#2a6fdb", lw=0.8, label="left")
    ax.plot(sec[win], track.right_stifle[win], color="#d9822b", lw=0.8, label="right")
    ax.set_ylabel("stifle ($^\\circ$)", fontsize=5.5, labelpad=1)
    ax.tick_params(labelsize=5, length=1.5, pad=1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
    ax.set_facecolor("white")
    ax.legend(fontsize=4.8, frameon=False, loc="upper right", ncol=2, handlelength=1, borderaxespad=0.1)
    text(x5 + (0.99 - x5) / 2 + 0.02, top - 0.012, "joint angles over 6 s,\nstride, metric size", size=cap)

    # ---- bottom: learning from all fits
    panel(0.005, 0.005, 0.99, 0.375, learn_bg)
    text(0.988, 0.37, "Learning from all fits (Sec. 3.5)", size=7, weight="bold", ha="right")
    by, bh = 0.035, 0.265
    specs = [("Pseudo-labels", "fits of WildDepth windows,\n6 species, split by snippet,\nweighted by fit IoU"),
             ("Pose VAE prior", "6D rotations + species\n$\\rightarrow$ 32-d latent; dropout;\nlatent smoothing on tracks"),
             ("Student", "frozen PRIMA ViT-H, BioCLIP;\nSMAL head fine-tuned;\nbone-scale + metric heads"),
             ("Losses", "labels, mask, LiDAR relief\n(point dropout), metric scale,\nprior, temporal; occlusion"),
             ("Image $\\rightarrow$ metric 3D", "one feed-forward pass,\nevaluated on held-out\nsnippets")]
    bw, gap = 0.172, 0.0275
    for k, (title, body) in enumerate(specs):
        bx = 0.012 + k * (bw + gap)
        box(bx, by, bw, bh, title, body)
        if k:
            arrow(bx - gap + 0.002, by + bh / 2, bx - 0.002, by + bh / 2)
    # every clip's fit feeds the labels: down from the fit, along the gap between the panels, into the first box
    from matplotlib.lines import Line2D
    xf, xl, yg = x4 + sq / 2, 0.012 + bw * 0.5, 0.39
    fig.lines.append(Line2D([xf, xf, xl], [top - 0.1, yg, yg], transform=fig.transFigure, color=ink, linewidth=0.7))
    arrow(xl, yg, xl, by + bh + 0.004)
    text(xf + 0.006, top - 0.1, "fits of all clips", size=5.4, color=muted, ha="left")
    target = OUT / "pipeline"
    fig.savefig(target / "pipeline.pdf", dpi=400)
    fig.savefig(target / "pipeline.png", dpi=200)
    fig.savefig(ROOT / "paper" / "fig" / "pipeline.pdf", dpi=400)
    print("wrote", target / "pipeline.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("figure", choices=("teaser", "pipeline"))
    args = parser.parse_args()
    {"teaser": teaser, "pipeline": pipeline}[args.figure]()


if __name__ == "__main__":
    main()
