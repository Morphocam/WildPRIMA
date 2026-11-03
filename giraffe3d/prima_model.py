"""Thin batch-inference wrapper around PRIMA, using its dataset crop and camera conventions."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# SMAL joint indices, from `name2id35` in prima/models/smal_wrapper.py.
_PELVIS, _LLEG1, _LFOOT, _NECK, _HEAD = 1, 7, 10, 15, 16


@dataclass
class Prediction:
    vertices: np.ndarray  # (N, V, 3) model space, root-relative
    cam_t: np.ndarray  # (N, 3) translation into the full-frame camera
    betas: np.ndarray  # (N, B)
    global_orient: np.ndarray  # (N, 3, 3)
    pose: np.ndarray  # (N, J, 3, 3)
    focal: float  # pixels, in the full frame
    shape_stats: list[dict]


class PrimaModel:
    def __init__(self, prima_dir: Path, device: str = "cuda"):
        from prima.models import load_prima
        from prima.utils.weights import resolve_prima_checkpoint_path

        prima_dir = Path(prima_dir).resolve()
        checkpoint = resolve_prima_checkpoint_path(
            str(prima_dir / "data/PRIMA/checkpoints/best_inference.ckpt"), data_dir=prima_dir / "data"
        )
        # The checkpoint's config refers to SMAL assets relative to the PRIMA repository root.
        previous_cwd = os.getcwd()
        os.chdir(prima_dir)
        try:
            self.model, self.cfg = load_prima(checkpoint)
        finally:
            os.chdir(previous_cwd)
        self.device = torch.device(device)
        self.model = self.model.to(self.device).eval()
        self.faces = self.model.smal.faces.cpu().numpy()

    def focal_for(self, width: int, height: int) -> float:
        return float(self.cfg.EXTRA.FOCAL_LENGTH) / float(self.cfg.MODEL.IMAGE_SIZE) * max(width, height)

    @torch.no_grad()
    def __call__(self, frame_bgr: np.ndarray, boxes: np.ndarray, batch_size: int = 8) -> Prediction:
        from prima.datasets.vitdet_dataset import ViTDetDataset
        from prima.utils import recursive_to
        from prima.utils.renderer import cam_crop_to_full

        dataset = ViTDetDataset(self.cfg, frame_bgr, boxes)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        height, width = frame_bgr.shape[:2]
        focal = self.focal_for(width, height)

        parts: dict[str, list[np.ndarray]] = {k: [] for k in ("v", "t", "b", "g", "p")}
        for batch in loader:
            batch = recursive_to(batch, self.device)
            out = self.model(batch)
            cam_t = cam_crop_to_full(
                out["pred_cam"], batch["box_center"].float(), batch["box_size"].float(),
                batch["img_size"].float(), focal,
            )
            params = out["pred_smal_params"]
            n = len(cam_t)
            parts["v"].append(out["pred_vertices"].cpu().numpy())
            parts["t"].append(cam_t.cpu().numpy())
            parts["b"].append(params["betas"].reshape(n, -1).cpu().numpy())
            parts["g"].append(params["global_orient"].reshape(n, 3, 3).cpu().numpy())
            parts["p"].append(params["pose"].reshape(n, -1, 3, 3).cpu().numpy())

        betas = np.concatenate(parts["b"])
        return Prediction(
            vertices=np.concatenate(parts["v"]),
            cam_t=np.concatenate(parts["t"]),
            betas=betas,
            global_orient=np.concatenate(parts["g"]),
            pose=np.concatenate(parts["p"]),
            focal=focal,
            shape_stats=self.shape_stats(betas),
        )

    @torch.no_grad()
    def shape_stats(self, betas: np.ndarray) -> list[dict]:
        """Body proportions from the shape alone (neutral pose), so they are independent of gait."""
        smal = self.model.smal
        b = torch.as_tensor(betas, device=self.device, dtype=smal.shapedirs.dtype)
        shaped = smal.v_template + torch.einsum("vcb,nb->nvc", smal.shapedirs[..., : b.shape[1]], b)
        joints = torch.einsum("jv,nvc->njc", smal.J_regressor, shaped)

        def bone(a: int, c: int) -> torch.Tensor:
            return (joints[:, a] - joints[:, c]).norm(dim=-1)

        neck, body, leg = bone(_NECK, _HEAD), bone(_PELVIS, _NECK), bone(_LLEG1, _LFOOT)
        return [
            {
                "neck_len": float(neck[i]),
                "body_len": float(body[i]),
                "leg_len": float(leg[i]),
                "neck_to_body": float(neck[i] / body[i]),
                "leg_to_body": float(leg[i] / body[i]),
            }
            for i in range(len(b))
        ]
