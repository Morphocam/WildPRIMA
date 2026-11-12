"""Step 3: a learned pose prior and latent smoothing, from the per-clip fits.

    python pose_prior.py collect      # fits -> new_output/dataset/poses.npz (+ split.json)
    python pose_prior.py train        # pose VAE -> new_output/prior/vae.pt
    python pose_prior.py smooth RUN.. # latent smoothing of a run's tracks, scored like the fit -> new_output/prior/smoothing/

Runs are the label windows in new_output/labels/runs/ plus the clips fitted earlier (outputs/, listed in EXISTING).
The split is by snippet (never by frame), fixed in new_output/dataset/split.json; the clips fitted earlier are training
data, since the method's settings were tuned on them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from smplx.lbs import batch_rodrigues

from animal3d.body import BODY_BOUNDS, AnimalBody, HornShape
from animal3d.prior import SPECIES_CLASSES, PoseVAE, rotation_error_deg, species_class, vae_loss

ROOT = Path(__file__).resolve().parent
NEW = ROOT / "new_output"
SCRATCH = Path(__file__).resolve().parent / "new_output" / "inputs"  # copies of inputs that lived in a temporary folder
# Earlier fits: run folder -> (input folder, calibration option)
EXISTING = {
    "giraffe_4_lidar": ("input_30s/giraffe_4", "auto"),
    "Ankle_cow_5_lidar": ("input_30s/Ankle_cow_5", "auto"),
    "Ankle_cow_2_lidar": ("input_30s/Ankle_cow_2_rgbd", "auto"),
    "Rhino_1_lidar": ("input_30s/Rhino_1", "auto"),
    "Rhino_3_lidar": ("input_30s/Rhino_3", "auto"),
    "Camel_1_domes": ("input_30s/Camel_1", "auto"),
    "Lion_1_lidar": ("input_30s/Lion_1", "input_30s/Ankle_cow_2_rgbd/calibration.json"),
}
# Whole snippets held out, one per species each (chosen before any training; Bubye Valley giraffe for a second site).
SPLIT = {
    "test": ["Ankle_cow_6", "camel_4", "Rhino_2", "19371340_giraffe_2", "lion_4", "donkey_5"],
    "val": ["Ankle_cow_3", "camel_3", "Rhino_4", "graffe_3", "lion_1", "donkey_2"],
}
MIN_IOU, MIN_BOX_H = 0.7, 40


def run_folders() -> dict[str, tuple[Path, Path, str]]:
    """name -> (run folder, input folder, calibration)."""
    runs = {}
    for out in sorted((NEW / "labels" / "runs").glob("*")):
        if (out / "fit.npz").exists():
            runs[out.name] = (out, NEW / "labels" / "inputs" / out.name, "auto")
    for name, (inp, calib) in EXISTING.items():
        if (ROOT / "outputs" / name / "fit.npz").exists():
            runs[name] = (ROOT / "outputs" / name, ROOT / inp if not inp.startswith("/") else Path(inp), calib)
    return runs


def split_of(name: str) -> str:
    if name in EXISTING:
        return "train"
    for part in ("test", "val"):
        for snippet in SPLIT[part]:
            # Window names are <release>_<snippet>_w<k>; a bare snippet name matches any release.
            if name.rsplit("_w", 1)[0].endswith(snippet):
                return part
    return "train"


def run_species(out: Path) -> str:
    summary = json.loads((out / "summary.json").read_text())
    info = summary.get("shape", {}).get("species", {})
    return info.get("species") or info.get("label") or summary["input"].get("species", "")


# ------------------------------------------------------------------------------------------- collect


def collect() -> None:
    rows, poses, orients = [], [], []
    for name, (out, _, _) in run_folders().items():
        fit, metrics = np.load(out / "fit.npz"), pd.read_csv(out / "metrics.csv")
        frames = pd.to_numeric(metrics["name"], errors="coerce")
        frames = frames.astype(int).to_numpy() if frames.notna().all() else fit["frame"]
        species = run_species(out)
        usable = (~metrics.truncated & ~metrics.overlapping & (metrics.iou >= MIN_IOU) & (metrics.box_h >= MIN_BOX_H)).to_numpy()
        for i in range(len(metrics)):
            rows.append({"run": name, "split": split_of(name), "species": species, "species_class": species_class(species),
                         "track_id": int(fit["track_id"][i]), "frame": int(frames[i]), "iou": float(metrics.iou[i]),
                         "box_h": float(metrics.box_h[i]), "usable": bool(usable[i])})
        poses.append(fit["pose"])
        orients.append(fit["global_orient"])
    table = pd.DataFrame(rows)
    target = NEW / "dataset"
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target / "poses.npz", pose=np.concatenate(poses).astype(np.float32), global_orient=np.concatenate(orients).astype(np.float32))
    table.to_csv(target / "instances.csv", index=False)
    (target / "split.json").write_text(json.dumps(SPLIT, indent=2))
    usable = table[table.usable]
    print(f"{len(table)} instances from {table.run.nunique()} runs; {len(usable)} usable")
    print(usable.groupby(["split", "species_class"]).size().unstack(fill_value=0).rename(columns=dict(enumerate(SPECIES_CLASSES))))


# --------------------------------------------------------------------------------------------- train


def train(steps: int, seeds: int, beta: float, device: str) -> None:
    data = np.load(NEW / "dataset" / "poses.npz")
    table = pd.read_csv(NEW / "dataset" / "instances.csv")
    keep = table.usable.to_numpy()
    pose = torch.as_tensor(data["pose"])
    species = torch.as_tensor(table.species_class.to_numpy())
    tr = np.where(keep & (table.split == "train").to_numpy())[0]
    va = np.where(keep & (table.split == "val").to_numpy())[0]
    if len(va) < 50:  # too few validation windows finished yet: hold out 10% of training tracks instead, for now
        tracks = table.loc[tr, ["run", "track_id"]].drop_duplicates().sample(frac=0.1, random_state=0)
        held = table.loc[tr].merge(tracks, on=["run", "track_id"], how="left", indicator=True)["_merge"].eq("both").to_numpy()
        va, tr = tr[held], tr[~held]
        print(f"Validation snippets not ready: using {len(tracks)} held-out training tracks ({len(va)} instances)")
    print(f"train {len(tr)}, val {len(va)} instances")
    pose_d, species_d = pose.to(device), species.to(device)
    out = NEW / "prior"
    out.mkdir(parents=True, exist_ok=True)
    best, log = None, []
    for seed in range(seeds):
        torch.manual_seed(seed)
        model = PoseVAE().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        tr_d = torch.as_tensor(tr, device=device)
        for step in range(steps):
            model.train()
            idx = tr_d[torch.randint(len(tr_d), (512,), device=device)]
            w = min(1.0, step / (steps * 0.3)) * beta  # KL annealing
            loss, parts = vae_loss(model, pose_d[idx], species_d[idx], w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            v = torch.as_tensor(va, device=device)
            recon, mu, _ = model(pose_d[v], species_d[v])
            err = rotation_error_deg(recon, pose_d[v])
            tr_s = tr_d[:4096]
            tr_err = rotation_error_deg(model(pose_d[tr_s], species_d[tr_s])[0], pose_d[tr_s])
        result = {"seed": seed, "val_joint_error_deg_mean": float(err.mean()), "val_joint_error_deg_p95": float(err.quantile(0.95)),
                  "train_joint_error_deg_mean": float(tr_err.mean()), "kl": parts["kl"]}
        log.append(result)
        print(result)
        if best is None or result["val_joint_error_deg_mean"] < best[0]:
            best = (result["val_joint_error_deg_mean"], model.state_dict(), seed)
    torch.save({"state": best[1], "seed": best[2], "beta": beta, "latent": 32}, out / "vae.pt")
    (out / "train_log.json").write_text(json.dumps({"train": len(tr), "val": len(va), "beta": beta, "steps": steps, "runs": log}, indent=2))
    sample_sheet(PoseVAE().to(device), best[1], out / "samples.png", device)


@torch.no_grad()
def sample_sheet(model: PoseVAE, state, path: Path, device: str) -> None:
    """Stick figures of 8 random prior samples per species, from the side, on SMAL's template skeleton."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from animal3d.kinematics import posed_joints
    from run import load_smal

    model.load_state_dict(state)
    model.eval()
    smal = load_smal(ROOT / "third_party/PRIMA")
    rest = smal["j_regressor"] @ np.asarray(smal["v_template"])
    parents = smal["parents"]
    species = [s for s in SPECIES_CLASSES if s != "other"]
    fig, axes = plt.subplots(len(species), 8, figsize=(16, 2 * len(species)))
    g = torch.Generator(device=device).manual_seed(0)
    for r, name in enumerate(species):
        z = torch.randn(8, model.latent, generator=g, device=device)
        poses = model.decode(z, torch.full((8,), SPECIES_CLASSES.index(name), device=device)).cpu().numpy()
        joints = posed_joints(rest, parents, poses)
        for c in range(8):
            ax = axes[r, c]
            for j in range(1, 33):
                p = parents[j]
                ax.plot(joints[c, [p, j], 0], joints[c, [p, j], 2], color="#444", lw=1.5)
            ax.set_aspect("equal")
            ax.axis("off")
        axes[r, 0].set_title(name, loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


# -------------------------------------------------------------------------------------------- smooth


def rebuild_body(out: Path, smal: dict, device: str) -> AnimalBody:
    """The run's fitted body, from fit.npz and summary.json."""
    fit, summary = np.load(out / "fit.npz"), json.loads((out / "summary.json").read_text())
    shape = summary["shape"]
    bounds = dict(BODY_BOUNDS)
    ears = shape.get("species", {}).get("ears")
    if ears is not None:
        bounds["ears"] = (ears, ears, 1.6 * ears, 1.6 * ears)
    horns = HornShape(**shape["horns"]) if shape.get("horns") else None
    humps = shape.get("humps") or []
    body = AnimalBody(**smal, betas=fit["betas"], bounds=bounds, horns=horns, humps=len(humps)).to(device)
    with torch.no_grad():
        body.log_scale.copy_(torch.as_tensor(fit["log_scale"]))
        body.offsets.copy_(torch.as_tensor(fit["offsets"]))
        if humps:
            body.humps.raw.copy_(torch.tensor([[h["position"], h["height"], h["length"], h["width"]] for h in humps]))
    gap = np.abs(body.rest()[0].detach().cpu().numpy()[: len(fit["rest_vertices"])] - fit["rest_vertices"]).max()
    if gap > 1e-3:
        print(f"  warning: rebuilt body differs from the fit by up to {gap:.4f} units")
    for p in body.parameters():
        p.requires_grad_(False)
    return body


def neighbours(track: np.ndarray, frame: np.ndarray, max_gap: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Consecutive pairs and triples (indices) along each track, allowing small gaps."""
    pairs, triples = [], []
    order = np.lexsort((frame, track))
    for a, b, c in zip(order, order[1:], order[2:]):
        if track[a] == track[b] == track[c] and frame[b] - frame[a] <= max_gap and frame[c] - frame[b] <= max_gap:
            triples.append((a, b, c))
    for a, b in zip(order, order[1:]):
        if track[a] == track[b] and frame[b] - frame[a] <= max_gap:
            pairs.append((a, b))
    return np.array(pairs).reshape(-1, 2), np.array(triples).reshape(-1, 3)


def joint_jitter(body: AnimalBody, pose: torch.Tensor, triples: np.ndarray) -> float:
    """Median joint acceleration in the body frame (model units per frame squared): articulation only."""
    if not len(triples):
        return float("nan")
    eye = torch.eye(3, device=pose.device).expand(len(pose), 3, 3)
    with torch.no_grad():
        j = torch.cat([body.posed_joints(eye[s:s + 256], pose[s:s + 256]) for s in range(0, len(pose), 256)])
    t = torch.as_tensor(triples, device=pose.device)
    acc = (j[t[:, 2]] - 2 * j[t[:, 1]] + j[t[:, 0]]).norm(dim=-1).mean(dim=1)
    return float(acc.median())


def smooth(names: list[str], iters: int, w_smooth: float, w_depth: float, device: str, tag: str = "", mode: str = "residual", lr: float = 1e-3) -> None:
    """mode 'residual': poses stay the fit's plus per-joint corrections, and the VAE's encoding of them is smoothed in time.
    mode 'decode': poses are the VAE's decoding of a latent code per frame (loses detail the VAE cannot reproduce)."""
    import run as R
    from animal3d.fit import depth_loss, mask_loss, project_to_crop
    from animal3d.sources import open_source

    ckpt = torch.load(NEW / "prior" / "vae.pt", map_location=device)
    vae = PoseVAE().to(device)
    vae.load_state_dict(ckpt["state"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    smal = R.load_smal(ROOT / "third_party/PRIMA")
    runs = run_folders()
    target = NEW / "prior" / "smoothing"
    target.mkdir(parents=True, exist_ok=True)
    for name in names:
        out, inp, calib = runs[name]
        print(f"== {name}")
        source = open_source(inp, None, calib)
        df, pred, masks, _ = R.load_prima_results(out)
        fit, metrics = np.load(out / "fit.npz"), pd.read_csv(out / "metrics.csv")
        body = rebuild_body(out, smal, device)
        inst = R.build_instances(pred, df, masks, source, device)
        if source.has_lidar:
            R.attach_lidar(inst, source, df, masks)
        n = len(df)
        frames = pd.to_numeric(metrics["name"], errors="coerce")
        frames = frames.astype(int).to_numpy() if frames.notna().all() else fit["frame"]
        pairs, triples = neighbours(fit["track_id"], frames)
        species = torch.full((n,), species_class(run_species(out)), device=device)
        pose0 = torch.as_tensor(fit["pose"], device=device)
        R0 = torch.as_tensor(fit["global_orient"], device=device)
        t0 = torch.as_tensor(fit["cam_t"], device=device)
        with torch.no_grad():
            z0 = vae.encode(pose0, species)[0]
            projected = vae.decode(z0, species)
        z = torch.nn.Parameter(z0.clone())
        d_pose = torch.nn.Parameter(torch.zeros(n, 34, 3, device=device))
        d_orient = torch.nn.Parameter(torch.zeros(n, 3, device=device))
        d_cam = torch.nn.Parameter(torch.zeros(n, 3, device=device))
        first = [z] if mode == "decode" else [d_pose]
        opt = torch.optim.Adam([{"params": first, "lr": lr if mode == "residual" else 0.01}, {"params": [d_orient, d_cam], "lr": 1e-3}])

        def pose_of(idx):
            if mode == "decode":
                return vae.decode(z[idx], species[idx])
            return batch_rodrigues(d_pose[idx].reshape(-1, 3)).reshape(len(idx), 34, 3, 3) @ pose0[idx]
        tri = torch.as_tensor(triples, device=device)
        g = torch.Generator(device=device).manual_seed(0)

        def posed(idx):
            pose = pose_of(idx)
            orient = batch_rodrigues(d_orient[idx]) @ R0[idx]
            t = t0[idx] + d_cam[idx] * t0[idx, 2:]
            return body(orient, pose), t

        for step in range(iters):
            idx = torch.randperm(n, generator=g, device=device)[:48]
            v, t = posed(idx)
            uv = project_to_crop(v, t, inst.camera[idx], inst.center[idx], inst.size[idx])
            outside, coverage = mask_loss(uv, inst.distance[idx], inst.samples[idx])
            loss = outside + coverage + 1e-1 * (d_orient.pow(2).sum(-1).mean() + d_cam.pow(2).sum(-1).mean())
            if w_depth and inst.lidar_d is not None:
                loss = loss + w_depth * depth_loss(v, t, uv, inst.lidar_uv[idx], inst.lidar_d[idx], inst.lidar_ok[idx])
            if mode == "decode":
                codes = z
                loss = loss + 1e-4 * z[idx].pow(2).sum(dim=1).mean()  # stay where the prior has mass
            else:
                all_idx = torch.arange(n, device=device)
                codes = vae.encode(pose_of(all_idx), species)[0]
                loss = loss + 1e-1 * d_pose.pow(2).sum(dim=-1).mean()  # all frames, so Adam momentum cannot drift unsampled ones
            if len(tri):
                loss = loss + w_smooth * (codes[tri[:, 2]] - 2 * codes[tri[:, 1]] + codes[tri[:, 0]]).pow(2).sum(dim=1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            all_idx = torch.arange(n, device=device)
            pose_s = torch.cat([pose_of(all_idx[s:s + 256]) for s in range(0, n, 256)])
            verts = torch.cat([posed(all_idx[s:s + 128])[0] for s in range(0, n, 128)]).cpu().numpy()
            cam_t = (t0 + d_cam * t0[:, 2:]).cpu().numpy()
        camera_for = lambda w, h: R.fit_camera(source, w, h)
        scored = R.score_meshes(source, df, masks, body.faces, verts, cam_t, camera_for)
        clean = (~df.truncated & ~df.overlapping).to_numpy()
        med = lambda frame, col: float(np.nanmedian(frame.loc[clean, col])) if col in frame else None
        result = {
            "run": name, "split": split_of(name), "mode": mode, "instances": n, "clean": int(clean.sum()), "iters": iters, "w_smooth": w_smooth, "lr": lr,
            "w_depth": w_depth, "vae_projection_error_deg": float(rotation_error_deg(projected, pose0).mean()),
            "iou": {"fit": med(metrics, "iou"), "smoothed": med(scored, "iou")},
            "relief_error_held_m": {"fit": med(metrics, "relief_error_held_m"), "smoothed": med(scored, "relief_error_held_m"),
                                    "flat": med(metrics, "flat_relief_held_m")},
            "jitter": {"fit": joint_jitter(body, pose0, triples), "vae_projection": joint_jitter(body, projected, triples),
                       "smoothed": joint_jitter(body, pose_s, triples)},
            "pose_change_deg": float(rotation_error_deg(pose_s, pose0).mean()),
        }
        print(json.dumps(result, indent=1))
        run_dir = target / (f"{name}_{tag}" if tag else name)
        run_dir.mkdir(exist_ok=True)
        (run_dir / "result.json").write_text(json.dumps(result, indent=2))
        np.savez_compressed(run_dir / "smoothed.npz", pose=pose_s.cpu().numpy(), global_orient=(batch_rodrigues(d_orient.detach()) @ R0).cpu().numpy(),
                            cam_t=cam_t, frame=frames, track_id=fit["track_id"], z=z.detach().cpu().numpy())
        scored.to_csv(run_dir / "metrics.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("collect")
    t = sub.add_parser("train")
    t.add_argument("--steps", type=int, default=6000)
    t.add_argument("--seeds", type=int, default=3)
    t.add_argument("--beta", type=float, default=1e-3)
    s = sub.add_parser("smooth")
    s.add_argument("runs", nargs="+")
    s.add_argument("--iters", type=int, default=1500)
    s.add_argument("--w-smooth", type=float, default=1e-2)
    s.add_argument("--w-depth", type=float, default=0.3)
    s.add_argument("--tag", default="", help="suffix for the output folder, to keep settings apart")
    s.add_argument("--mode", choices=("residual", "decode"), default="residual")
    s.add_argument("--lr", type=float, default=1e-3, help="residual-mode learning rate")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.cmd == "collect":
        collect()
    elif args.cmd == "train":
        train(args.steps, args.seeds, args.beta, device)
    else:
        smooth(args.runs, args.iters, args.w_smooth, args.w_depth, device, args.tag, args.mode, args.lr)


if __name__ == "__main__":
    main()
