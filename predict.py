"""Single-image metric 3D animals with the released WildPRIMA student.

    python predict.py IMAGE [IMAGE ...] [--weights DIR_OR_HF_REPO] [--calibration none|longleat_v2|bubye|calib.json] [--out DIR]

For every animal Mask R-CNN finds, writes <image>_animalNN.obj (camera frame, metres) and one <image>.json with the
box, detector class, distance, metres per model unit and body size. Needs the PRIMA checkpoint and SMAL files in
third_party/PRIMA (see README), because the frozen ViT-H/BioCLIP backbone and SMAL are not redistributed with the
student weights. Horns and humps are not predicted by the student; the per-clip fitting (run.py) adds them.

Without a calibration the camera is PRIMA's assumed one (focal = 3.9 x the longer image side), so metric outputs are
only as good as that assumption; pass the real intrinsics for metric use.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
FORMAT = "wildprima-student/1"


def resolve_weights(spec: str) -> Path:
    path = Path(spec)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download  # a Hugging Face repo id, e.g. "<user>/wildprima-student"
    return Path(snapshot_download(spec, allow_patterns=["*.json", "*.safetensors", "*.md", "LICENSE*"]))


def load_student(weights: Path, prima, device: str):
    from safetensors.torch import load_file
    from student import Student

    config = json.loads((weights / "config.json").read_text())
    if config.get("format") != FORMAT:
        raise SystemExit(f"{weights}: expected format {FORMAT}, found {config.get('format')}")
    student = Student(prima.model.smal_head, dim=config["token_dim"], dropout=config["dropout"]).to(device)
    missing, unexpected = student.load_state_dict(load_file(str(weights / "model.safetensors"), device=device), strict=False)
    if missing or unexpected:
        raise SystemExit(f"weights do not match the model: missing {missing[:5]}, unexpected {unexpected[:5]}")
    return student.eval(), config


def camera_for(spec: str, width: int, height: int):
    import run as R
    from animal3d.sources import load_calibration, scaled_camera

    if spec == "none":
        return R.prima_camera(width, height)
    calibration = load_calibration(spec, None, {})
    return scaled_camera(calibration, width, height)


@torch.no_grad()
def predict(images: list[Path], weights: Path, calibration: str, out: Path, device: str) -> None:
    import run as R
    from animal3d.detection import AnimalDetector, COCO_ANIMALS
    from student import BatchedBody, camera_translation, load_prima, prima_tokens

    prima = load_prima(device)
    student, config = load_student(weights, prima, device)
    smal = R.load_smal(ROOT / "third_party/PRIMA")
    body = BatchedBody(smal).to(device)
    faces = np.asarray(smal["faces"], dtype=np.int64)
    detector = AnimalDetector(device=device)
    out.mkdir(parents=True, exist_ok=True)
    for path in images:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"cannot read {path}")
        h, w = bgr.shape[:2]
        dets, _ = detector(bgr)
        result = {"image": str(path), "weights": config["name"], "calibration": calibration, "animals": []}
        if len(dets):
            got = prima_tokens(prima, bgr, dets.boxes)
            cam = camera_for(calibration, w, h)
            n = len(dets)
            t = lambda x: torch.as_tensor(np.asarray(x, np.float32), device=device)
            meta = {"box_center": t(got["box_center"]).reshape(n, 2), "box_size": t(got["box_size"]).reshape(n),
                    "img_size": t(got["img_size"]).reshape(n, 2), "prima_focal": t([R.PRIMA_FOCAL_RATIO * max(w, h)] * n),
                    "camera": t([[cam.fx, cam.fy, cam.cx, cam.cy]] * n)}
            o = student(torch.as_tensor(got["tokens"], device=device).float())
            idx = torch.arange(n, device=device)
            trans = camera_translation(o["cam"], meta, idx)
            verts = body(o["betas"], o["log_scale"], o["orient"], o["pose"])
            body_frame = body(o["betas"], o["log_scale"], torch.eye(3, device=device).expand(n, 3, 3), o["pose"])
            mpu = o["log_mpu"].exp()
            for k in range(n):
                m = float(mpu[k])
                metric = ((verts[k] + trans[k]) * m).cpu().numpy()
                extent = (body_frame[k].max(0).values - body_frame[k].min(0).values).cpu().numpy() * m
                name = f"{path.stem}_animal{k + 1:02d}.obj"
                with open(out / name, "w") as f:
                    f.write("".join(f"v {x:.5f} {y:.5f} {z:.5f}\n" for x, y, z in metric))
                    f.write("".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in faces))
                result["animals"].append({
                    "mesh": name, "box_xyxy": [float(v) for v in dets.boxes[k]], "score": float(dets.scores[k]),
                    "coco_class": COCO_ANIMALS.get(int(dets.classes[k]), str(int(dets.classes[k]))),
                    "distance_m": float(trans[k, 2]) * m, "metres_per_unit": m,
                    "body_length_m": float(extent[0]), "body_height_m": float(extent[2]),
                })
        (out / f"{path.stem}.json").write_text(json.dumps(result, indent=2))
        print(f"{path.name}: {len(result['animals'])} animal(s) -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--weights", default=str(ROOT / "release" / "huggingface"), help="release folder or Hugging Face repo id")
    parser.add_argument("--calibration", default="none", help="none, longleat_v2, bubye, or a calibration.json")
    parser.add_argument("--out", type=Path, default=Path("predictions"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = args.device
    predict(args.images, resolve_weights(args.weights), args.calibration, args.out, device)


if __name__ == "__main__":
    main()
