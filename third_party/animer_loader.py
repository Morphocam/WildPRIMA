"""Loads the released AniMer (Lyu et al., CVPR 2025) from third_party/AniMer (Hugging Face Space code + checkpoint).

pytorch3d is only imported by AniMer's training losses; a stub stands in so inference runs in the prima environment.
"""
import os
import sys
import types
from pathlib import Path

ANIMER = Path(__file__).resolve().parent / "AniMer"


def load_animer(device="cuda"):
    if "pytorch3d" not in sys.modules:
        p3, tr = types.ModuleType("pytorch3d"), types.ModuleType("pytorch3d.transforms")

        def _training_only(*_, **__):
            raise NotImplementedError("pytorch3d is not installed; only needed for AniMer training")

        tr.matrix_to_axis_angle = tr.axis_angle_to_matrix = _training_only
        p3.transforms = tr
        sys.modules["pytorch3d"], sys.modules["pytorch3d.transforms"] = p3, tr
    sys.path.insert(0, str(ANIMER))
    cwd = os.getcwd()
    os.chdir(ANIMER)  # the config's SMAL path is relative
    try:
        from amr.configs import get_config
        from amr.models import AMR

        cfg = get_config(str(ANIMER / "data/AniMer/config.yaml"))
        model = AMR.load_from_checkpoint(str(ANIMER / "data/AniMer/checkpoints/checkpoint.ckpt"), map_location="cpu",
                                         cfg=cfg, strict=False, weights_only=True)
    finally:
        os.chdir(cwd)
    return model.to(device).eval(), cfg
