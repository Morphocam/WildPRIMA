"""Step 4: a feed-forward student that predicts the extended body from one crop, trained on the per-clip fits.

    python student.py cache [RUN ...]   # frozen PRIMA backbone tokens + targets per run -> new_output/student/cache/
    python student.py train [--hours 12] # -> new_output/student/student.pt, train_log.jsonl
    python student.py eval [RUN ...]     # -> new_output/eval/

Model: PRIMA's own SMAL decoder (initialised from its checkpoint) on the frozen ViT-H and BioCLIP tokens, plus a small
attention-pooled head for bone scales (13 groups x length, thickness) and the animal's metres per model unit, which with
the camera gives its distance and size in metres. Horns and humps are not predicted: at inference they come from the
species table, as in the fits.

Losses: pseudo-labels from the fits (pose, orientation, bone scales, translation), weighted by fit quality; the fits'
mask term on the predicted mesh; the LiDAR relief term on the fitting half of the points only; metres per unit
against the track's LiDAR scale; the pose VAE as a prior; and latent closeness of neighbouring frames.
Regularisation: dropout 0.1 in the new head, patch-token dropout in blocks (occlusion), LiDAR point dropout.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from smplx.lbs import batch_rigid_transform
from torch import nn

from animal3d.body import BODY_BOUNDS, BONE_GROUPS, _PRIMARY_CHILD
from animal3d.prior import PoseVAE, species_class
from pose_prior import NEW, ROOT, rebuild_body, run_folders, run_species, split_of

CACHE = NEW / "student" / "cache"
TOKENS = 193  # 12 x 16 ViT-H patch tokens + 1 BioCLIP token
GROUPS = list(BONE_GROUPS)
LOG_LOW = torch.log(torch.tensor([[BODY_BOUNDS[g][0], BODY_BOUNDS[g][2]] for g in GROUPS]))
LOG_HIGH = torch.log(torch.tensor([[BODY_BOUNDS[g][1], BODY_BOUNDS[g][3]] for g in GROUPS]))


# --------------------------------------------------------------------------------------------- cache


def load_prima(device):
    from animal3d.prima_model import PrimaModel

    return PrimaModel(ROOT / "third_party/PRIMA", device)


@torch.no_grad()
def prima_tokens(prima, frame_bgr, boxes):
    """Tokens the SMAL head sees, and the crop geometry, for each box, exactly as PRIMA's forward_step makes them."""
    from prima.datasets.vitdet_dataset import ViTDetDataset
    from prima.utils import recursive_to

    model = prima.model
    loader = torch.utils.data.DataLoader(ViTDetDataset(prima.cfg, frame_bgr, boxes), batch_size=8, shuffle=False, num_workers=0)
    out = {"tokens": [], "box_center": [], "box_size": [], "img_size": []}
    for batch in loader:
        batch = recursive_to(batch, prima.device)
        x = batch["img"]
        feats, _ = model.backbone(x[:, :, :, 32:-32])
        if feats.ndim == 4:
            b, d, h, w = feats.shape
            feats = feats.permute(0, 2, 3, 1).reshape(b, h * w, d)
        if model.bioclip_embedding is not None:
            feats = torch.cat([feats, model.bioclip_embedding(x)[:, None]], dim=1)
        out["tokens"].append(feats.half().cpu().numpy())
        for k in ("box_center", "box_size", "img_size"):
            out[k].append(batch[k].float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def cache_run(name, prima, smal, device) -> None:
    import run as R
    from animal3d.sources import open_source

    out, inp, calib = run_folders()[name]
    target = CACHE / name
    if (target / "meta.npz").exists():
        return
    target.mkdir(parents=True, exist_ok=True)
    source = open_source(inp, None, calib)
    df, pred, masks, _ = R.load_prima_results(out)
    fit, metrics = np.load(out / "fit.npz"), pd.read_csv(out / "metrics.csv")
    n = len(df)
    tokens = np.lib.format.open_memmap(target / "tokens.npy", mode="w+", dtype=np.float16, shape=(n, TOKENS, 1280))
    geo = {k: np.zeros((n, 2) if k != "box_size" else (n,), np.float32) for k in ("box_center", "box_size", "img_size")}
    by_frame = {f: g.index.to_numpy() for f, g in df.groupby("frame")}
    for frame in source:
        rows = by_frame.get(frame.index)
        if rows is None:
            continue
        got = prima_tokens(prima, frame.bgr, df.loc[rows, ["x1", "y1", "x2", "y2"]].to_numpy(np.float32))
        tokens[rows] = got["tokens"]
        for k in geo:
            geo[k][rows] = got[k].reshape(geo[k][rows].shape)
    tokens.flush()

    inst = R.build_instances(pred, df, masks, source, device)
    if source.has_lidar:
        R.attach_lidar(inst, source, df, masks)
    usable = (~metrics.truncated & ~metrics.overlapping & (metrics.iou >= 0.7) & (metrics.box_h >= 40)).to_numpy()
    mpu = metrics["metres_per_unit"] if "metres_per_unit" in metrics else pd.Series(np.nan, index=metrics.index)
    track_mpu = mpu.groupby(metrics.track_id).transform("median").to_numpy()
    frames = pd.to_numeric(metrics["name"], errors="coerce")
    frames = frames.astype(int).to_numpy() if frames.notna().all() else fit["frame"]
    np.savez_compressed(
        target / "meta.npz", **geo,
        prima_focal=np.array([R.PRIMA_FOCAL_RATIO * max(w, h) for w, h in zip(df.width, df.height)], np.float32),
        camera=inst.camera.cpu().numpy(), center=inst.center.cpu().numpy(), size=inst.size.cpu().numpy(),
        distance=inst.distance.cpu().numpy().astype(np.float16), samples=inst.samples.cpu().numpy().astype(np.float16),
        lidar_uv=(inst.lidar_uv.cpu().numpy() if inst.lidar_uv is not None else np.zeros((n, 0, 2), np.float32)).astype(np.float16),
        lidar_d=inst.lidar_d.cpu().numpy() if inst.lidar_d is not None else np.zeros((n, 0), np.float32),
        lidar_ok=inst.lidar_ok.cpu().numpy() if inst.lidar_ok is not None else np.zeros((n, 0), bool),
        pose=fit["pose"], orient=fit["global_orient"], cam_t=fit["cam_t"], log_scale=fit["log_scale"],
        prima_betas=pred["betas"], iou=metrics.iou.to_numpy(np.float32), usable=usable, track=fit["track_id"], frame=frames,
        mpu=track_mpu.astype(np.float32), lidar_depth=metrics["lidar_depth_m"].to_numpy(np.float32),
        species=np.full(n, species_class(run_species(out))), split=np.array(split_of(name)),
    )
    print(f"cached {name}: {n} instances, {usable.sum()} usable")


# --------------------------------------------------------------------------------------------- model


class BatchedBody(nn.Module):
    """AnimalBody's skeleton stretching and skinning, with bone scales and betas per sample (no offsets or parts)."""

    def __init__(self, smal: dict):
        super().__init__()
        f = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)
        self.register_buffer("v_template", f(smal["v_template"]))
        self.register_buffer("shapedirs", f(smal["shapedirs"]))
        self.register_buffer("j_regressor", f(smal["j_regressor"]))
        self.register_buffer("weights", f(smal["weights"]))
        self.parents = np.asarray(smal["parents"]).astype(np.int64)
        self.register_buffer("parents_t", torch.as_tensor(self.parents))
        n = len(self.parents)
        group = torch.zeros(n, dtype=torch.long)
        for g, name in enumerate(GROUPS):
            group[list(BONE_GROUPS[name])] = g
        self.register_buffer("group", group)
        children = {j: [] for j in range(n)}
        for j in range(1, n):
            children[int(self.parents[j])].append(j)
        self.child = [_PRIMARY_CHILD.get(j, children[j][0] if len(children[j]) == 1 else None) for j in range(n)]

    def forward(self, betas, log_scale, orient, pose):
        """betas (B,41), log_scale (B,13,2), orient (B,3,3), pose (B,34,3,3) -> vertices (B,V,3) in the body frame."""
        B, n = len(betas), len(self.parents)
        shaped = self.v_template + torch.einsum("vcb,nb->nvc", self.shapedirs, betas)
        joints = torch.einsum("jv,nvc->njc", self.j_regressor, shaped)
        dirs = []
        for j in range(n):
            c = self.child[j]
            if c is not None:
                d = joints[:, c] - joints[:, j]
                dirs.append(d / d.norm(dim=-1, keepdim=True).clamp(min=1e-8))
            elif j > 0:
                dirs.append(dirs[int(self.parents[j])])
            else:
                dirs.append(torch.tensor([1.0, 0.0, 0.0], device=betas.device).expand(B, 3))
        dirs = torch.stack(dirs, dim=1)  # (B,J,3)
        scale = torch.exp(torch.max(torch.min(log_scale, LOG_HIGH.to(betas.device)), LOG_LOW.to(betas.device)))[:, self.group]
        length, thick = scale[..., 0, None, None], scale[..., 1, None, None]
        stretch = thick * torch.eye(3, device=betas.device) + (length - thick) * dirs[..., :, None] * dirs[..., None, :]
        new = [joints[:, 0]]
        for j in range(1, n):
            p = int(self.parents[j])
            new.append(new[p] + (stretch[:, p] @ (joints[:, j] - joints[:, p])[..., None])[..., 0])
        new = torch.stack(new, dim=1)
        rel = shaped[:, :, None] - joints[:, None]  # (B,V,J,3)
        moved = new[:, None] + torch.einsum("bjxy,bvjy->bvjx", stretch, rel)
        rest = (self.weights[None, :, :, None] * moved).sum(dim=2)
        rotations = torch.cat([orient[:, None], pose], dim=1)
        _, transforms = batch_rigid_transform(rotations, new, self.parents_t)
        blended = torch.einsum("vj,bjxy->bvxy", self.weights, transforms)
        homo = torch.cat([rest, torch.ones_like(rest[..., :1])], dim=-1)
        return (blended @ homo[..., None])[..., :3, 0]


class Student(nn.Module):
    def __init__(self, smal_head: nn.Module, dim: int = 1280, dropout: float = 0.1):
        super().__init__()
        self.smal_head = smal_head
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        self.pool = nn.MultiheadAttention(dim, 8, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 512), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(512, len(GROUPS) * 2 + 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, tokens):
        params, cam, _ = self.smal_head(tokens)
        B = len(tokens)
        pooled = self.pool(self.query.expand(B, -1, -1), tokens, tokens, need_weights=False)[0][:, 0]
        extra = self.head(pooled)
        return {"orient": params["global_orient"].reshape(B, 3, 3), "pose": params["pose"].reshape(B, -1, 3, 3),
                "betas": params["betas"].reshape(B, -1), "cam": cam,
                "log_scale": extra[:, :-1].view(B, len(GROUPS), 2), "log_mpu": extra[:, -1]}


def camera_translation(cam, meta, idx):
    """PRIMA's crop camera -> full-frame translation (cam_crop_to_full) -> the real camera (move_to_camera), in torch."""
    c, s, img, f = meta["box_center"][idx], meta["box_size"][idx], meta["img_size"][idx], meta["prima_focal"][idx]
    bs = s * cam[:, 0] + 1e-9
    tz = 2 * f / bs
    tx = 2 * (c[:, 0] - img[:, 0] / 2) / bs + cam[:, 1]
    ty = 2 * (c[:, 1] - img[:, 1] / 2) / bs + cam[:, 2]
    k = meta["camera"][idx]
    depth = tz * (k[:, 0] + k[:, 1]) / 2 / f
    u = f * tx / tz + img[:, 0] / 2
    v = f * ty / tz + img[:, 1] / 2
    return torch.stack([depth * (u - k[:, 2]) / k[:, 0], depth * (v - k[:, 3]) / k[:, 1], depth], dim=1)


# ---------------------------------------------------------------------------------------------- data


class Data:
    """All cached runs of one split, in memory: tokens as fp16 on the CPU, the rest on the GPU."""

    def __init__(self, names, device, usable_only=True):
        parts, metas = [], []
        for name in names:
            meta = dict(np.load(CACHE / name / "meta.npz"))
            keep = np.where(meta["usable"])[0] if usable_only else np.arange(len(meta["usable"]))
            if not len(keep):
                continue
            parts.append(torch.from_numpy(np.load(CACHE / name / "tokens.npy", mmap_mode="r")[keep]))
            n_all = len(meta["usable"])
            meta["log_scale"] = np.broadcast_to(meta["log_scale"], (n_all,) + meta["log_scale"].shape).copy()  # one per clip
            meta = {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim and len(v) == n_all else v) for k, v in meta.items()}
            meta["run"] = np.full(len(keep), len(metas))
            metas.append(meta)
        self.names = names
        self.n = 0
        if not metas:
            return
        self.tokens = torch.cat(parts)
        L = max([m["lidar_d"].shape[1] for m in metas] + [1])
        merged = {}
        for k in metas[0]:
            if k == "split":
                continue
            vals = []
            for m in metas:
                v = m[k]
                if k in ("lidar_uv", "lidar_d", "lidar_ok") and v.shape[1] < L:
                    v = np.concatenate([v, np.zeros((len(v), L - v.shape[1]) + v.shape[2:], v.dtype)], axis=1)
                vals.append(v)
            merged[k] = np.concatenate(vals)
        self.meta = {k: torch.as_tensor(v.astype(np.float32) if v.dtype == np.float16 else v, device=device) for k, v in merged.items()}
        self.n = len(self.tokens)
        # The next frame of the same track (within 2 frames), for the temporal term; -1 when there is none.
        nxt = np.full(self.n, -1)
        run, track, frame = (merged[k] for k in ("run", "track", "frame"))
        order = np.lexsort((frame, track, run))
        for a, b in zip(order, order[1:]):
            if run[a] == run[b] and track[a] == track[b] and 0 < frame[b] - frame[a] <= 2:
                nxt[a] = b
        self.next = torch.as_tensor(nxt, device=device)


def occlude(tokens, p=0.3, size=(4, 6)):
    """Zero a random block of patch tokens (12 x 16 grid) in a fraction p of samples: occlusion on cached features."""
    B = len(tokens)
    hit = torch.rand(B, device=tokens.device) < p
    if not hit.any():
        return tokens
    grid = tokens[:, :192].view(B, 12, 16, -1)
    h = torch.randint(2, size[0] + 1, (1,)).item()
    w = torch.randint(2, size[1] + 1, (1,)).item()
    y = torch.randint(0, 12 - h + 1, (B,), device=tokens.device)
    x = torch.randint(0, 16 - w + 1, (B,), device=tokens.device)
    rows = torch.arange(12, device=tokens.device)[None, :, None]
    cols = torch.arange(16, device=tokens.device)[None, None, :]
    block = (rows >= y[:, None, None]) & (rows < (y + h)[:, None, None]) & (cols >= x[:, None, None]) & (cols < (x + w)[:, None, None])
    block &= hit[:, None, None]
    grid = grid.masked_fill(block[..., None], 0.0)
    return torch.cat([grid.view(B, 192, -1), tokens[:, 192:]], dim=1)


# --------------------------------------------------------------------------------------------- train


# The mask and depth terms are ~1e-3 in size against ~3e-2 for the pseudo-label pose term (first test run), so they are
# scaled up to weigh about as much; depth keeps the fits' 0.3 ratio to the mask term.
W = {"pose": 1.0, "orient": 1.0, "scale": 1.0, "trans": 1.0, "mask": 20.0, "depth": 6.0, "mpu": 0.3, "prior": 1e-3, "temporal": 0.1}


def losses(student, body, vae, data, idx, train, w=W):
    from animal3d.fit import depth_loss, mask_loss, project_to_crop

    m = data.meta
    tok = data.tokens[idx.cpu()].to(idx.device).float()
    if train:
        tok = occlude(tok)
    out = student(tok)
    quality = ((m["iou"][idx] - 0.6) / 0.3).clamp(0, 1)
    geo = lambda a, b: (a - b).pow(2).sum(dim=(-1, -2))
    L = {}
    L["pose"] = (geo(out["pose"], m["pose"][idx]).mean(dim=1) * quality).mean()
    L["orient"] = (geo(out["orient"], m["orient"][idx]) * quality).mean()
    L["scale"] = ((out["log_scale"] - m["log_scale"][idx]).abs().mean(dim=(1, 2)) * quality).mean()
    t = camera_translation(out["cam"], m, idx)
    t_fit = m["cam_t"][idx]
    L["trans"] = ((t - t_fit).abs().sum(dim=1) / t_fit[:, 2].abs().clamp(min=1e-3) * quality).mean()
    verts = body(out["betas"], out["log_scale"], out["orient"], out["pose"])
    uv = project_to_crop(verts, t, m["camera"][idx], m["center"][idx], m["size"][idx])
    outside, coverage = mask_loss(uv, m["distance"][idx], m["samples"][idx])
    L["mask"] = outside + coverage
    if m["lidar_ok"].shape[1]:
        ok = m["lidar_ok"][idx]
        if train:  # LiDAR point dropout: keep 30-100% of the points
            keep = torch.rand(len(idx), 1, device=idx.device) * 0.7 + 0.3
            ok = ok & (torch.rand(ok.shape, device=idx.device) < keep)
        L["depth"] = depth_loss(verts, t, uv, m["lidar_uv"][idx], m["lidar_d"][idx], ok)
        mpu = m["mpu"][idx]
        has = torch.isfinite(mpu) & (mpu > 0)
        L["mpu"] = (out["log_mpu"][has] - mpu[has].log()).abs().mean() if has.any() else out["log_mpu"].sum() * 0
    sp = m["species"][idx].long()
    mu = vae.encode(out["pose"], sp)[0]
    L["prior"] = mu.pow(2).sum(dim=1).mean()
    if train:
        nxt = data.next[idx]
        has = nxt >= 0
        if has.any():
            tok2 = data.tokens[nxt[has].cpu()].to(idx.device).float()
            mu2 = vae.encode(student(tok2)["pose"], sp[has])[0]
            L["temporal"] = (mu[has] - mu2).pow(2).sum(dim=1).mean()
    total = sum(w[k] * v for k, v in L.items())
    return total, {k: float(v) for k, v in L.items()}, out, t, verts


def train(hours: float, batch: int, device: str) -> None:
    from run import load_smal

    names = sorted(p.name for p in CACHE.iterdir() if (p / "meta.npz").exists())
    split = {n: str(np.load(CACHE / n / "meta.npz")["split"]) for n in names}
    tr = Data([n for n in names if split[n] == "train"], device)
    va = Data([n for n in names if split[n] == "val"], device)
    print(f"train {tr.n} instances from {len(tr.names)} runs; val {va.n} from {len(va.names)} runs")
    prima = load_prima(device)
    student = Student(prima.model.smal_head).to(device)
    del prima
    body = BatchedBody(load_smal(ROOT / "third_party/PRIMA")).to(device)
    ckpt = torch.load(NEW / "prior" / "vae.pt", map_location=device)
    vae = PoseVAE().to(device)
    vae.load_state_dict(ckpt["state"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    head = [p for n, p in student.named_parameters() if not n.startswith("smal_head.")]
    opt = torch.optim.AdamW([{"params": student.smal_head.parameters(), "lr": 1e-5}, {"params": head, "lr": 1e-4}], weight_decay=1e-4)
    out_dir = NEW / "student"
    log = (out_dir / "train_log.jsonl").open("a")
    start, step, best = time.time(), 0, float("inf")
    budget = hours * 3600
    epoch_steps = max(tr.n // batch, 50)
    while time.time() - start < budget:
        student.train()
        for _ in range(epoch_steps):
            frac = (time.time() - start) / budget
            for g, base in zip(opt.param_groups, (1e-5, 1e-4)):
                g["lr"] = base * 0.5 * (1 + np.cos(np.pi * min(frac, 1.0)))  # cosine over the time budget
            idx = torch.randint(tr.n, (batch,), device=device)
            loss, parts, *_ = losses(student, body, vae, tr, idx, train=True)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            step += 1
            if time.time() - start > budget:
                break
        val = evaluate_loss(student, body, vae, va if va.n else tr, batch)
        record = {"step": step, "hours": (time.time() - start) / 3600, "train": parts, "val": val}
        log.write(json.dumps(record) + "\n")
        log.flush()
        score = val["mask"] + 0.3 * val.get("depth", 0.0) + 0.1 * val["pose"]
        print(f"step {step} ({record['hours']:.2f} h): val mask {val['mask']:.5f} depth {val.get('depth', 0):.5f} pose {val['pose']:.4f} mpu {val.get('mpu', 0):.3f}", flush=True)
        if score < best:
            best = score
            torch.save({"state": student.state_dict(), "step": step, "val": val}, out_dir / "student.pt")


@torch.no_grad()
def evaluate_loss(student, body, vae, data, batch) -> dict:
    student.eval()
    sums, count = {}, 0
    for s in range(0, data.n, batch):
        idx = torch.arange(s, min(s + batch, data.n), device=data.meta["iou"].device)
        _, parts, *_ = losses(student, body, vae, data, idx, train=False)
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + v * len(idx)
        count += len(idx)
    return {k: v / count for k, v in sums.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache")
    c.add_argument("runs", nargs="*")
    t = sub.add_parser("train")
    t.add_argument("--hours", type=float, default=12.0)
    t.add_argument("--batch", type=int, default=32)
    e = sub.add_parser("eval")
    e.add_argument("runs", nargs="*")
    args = parser.parse_args()
    device = "cuda"
    (NEW / "student").mkdir(parents=True, exist_ok=True)
    if args.cmd == "cache":
        from run import load_smal

        prima, smal = load_prima(device), load_smal(ROOT / "third_party/PRIMA")
        for name in args.runs or list(run_folders()):
            cache_run(name, prima, smal, device)
    elif args.cmd == "train":
        train(args.hours, args.batch, device)
    else:
        from student_eval import evaluate

        evaluate(args.runs, device)


if __name__ == "__main__":
    main()
