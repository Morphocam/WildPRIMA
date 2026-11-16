"""Package the trained student and pose prior for release, separately from the experiment outputs.

    python export_release.py [--version 0.1.0] [--student new_output/student/student.pt] [--out release]

release/huggingface/  model repo: model.safetensors (student), pose_prior.safetensors (VAE), config.json,
                      README.md (model card with results), LICENSE (PRIMA's academic terms, which the weights inherit),
                      results/ (evaluation and training logs), SHA256SUMS
release/github/       code repo: the package, scripts, tests, predict.py, README.md, LICENSE (MIT, code only), NOTICE

Nothing is uploaded. Not included, by licence: PRIMA's ViT-H/BioCLIP backbone and checkpoint (fetched by setup_env.sh)
and the SMAL model files (MPI licence; downloaded with PRIMA's data).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
NEW = ROOT / "new_output"
FORMAT = "wildprima-student/1"
TEASER = ROOT / "paper" / "fig" / "teaser.jpg"  # rendered by `python make_figures.py teaser`
TEASER_CAPTION = "PRIMA (blue) against WildPRIMA (orange) on WildDepth giraffe, white rhinoceros, Bactrian camel and Ankole-Watusi cattle; numbers are silhouette IoU."
CODE = ["animal3d", "tests", "docs", "assets", "run.py", "student.py", "student_eval.py", "pose_prior.py", "kinematics.py",
        "make_labels.py", "make_figures.py", "paper_results.py", "predict.py", "export_release.py", "app.py", "run_step4.sh",
        "setup_env.sh", "requirements.txt", "student_refine.py", "baseline_animer.py", "sizes.py", "click_keypoints.py",
        "analyse_clicks.py", "third_party/animer_loader.py", "third_party/AniMer_SOURCE.txt"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_safetensors(state: dict, path: Path, metadata: dict) -> int:
    from safetensors.torch import save_file
    tensors = {k: v.detach().cpu().contiguous().float() if v.is_floating_point() else v.detach().cpu().contiguous()
               for k, v in state.items()}
    save_file(tensors, str(path), metadata={k: str(v) for k, v in metadata.items()})
    path.chmod(0o644)
    return sum(v.numel() for v in tensors.values())


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def results_table(summary: dict | None) -> str:
    if not summary:
        return "_Evaluation not yet run._\n"
    rows = ["| Split | Snippet | N | IoU PRIMA | IoU fit | IoU student | LiDAR PRIMA (m) | LiDAR fit (m) | LiDAR student (m) | Distance rel. err |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda v, d=2: "–" if v is None else f"{v:.{d}f}"
    for r in summary["runs"]:
        rows.append(f"| {r['split']} | {r['run']} | {r['clean']} | {f(r['iou_prima'])} | {f(r['iou_fit'])} | {f(r['iou_student'])} | "
                    f"{f(r['lidar_held_prima'], 3)} | {f(r['lidar_held_fit'], 3)} | {f(r['lidar_held_student'], 3)} | {f(r['distance_rel_error_student'])} |")
    return "\n".join(rows) + "\n"


def depth_ablation_table() -> str:
    """Per-clip fits with and without the LiDAR relief term (new_output/ablation_depth, paper Table tab:depth)."""
    path = NEW / "ablation_depth" / "summary.csv"
    if not path.exists():
        return ""
    import csv
    rows = list(csv.DictReader(path.open()))
    f = lambda v, d: f"{float(v):.{d}f}"
    out = ["## Per-clip fitting: the LiDAR relief term", "",
           "Same detections and PRIMA results, fitted from masks only (`--depth-weight 0`) or with LiDAR relief (`0.3`, default).",
           "Error is on LiDAR points held out from the fit (`results/depth_ablation.csv`).", "",
           "| Clip | IoU PRIMA | IoU masks only | IoU with LiDAR | LiDAR PRIMA (m) | Flat (m) | LiDAR masks only (m) | LiDAR with LiDAR (m) |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['clip']} | {f(r['prima_iou'], 2)} | {f(r['iou_w0'], 2)} | {f(r['iou_w03'], 2)} | {f(r['prima_held'], 3)} | "
                   f"{f(r['flat_held'], 3)} | {f(r['held_w0'], 3)} | {f(r['held_w03'], 3)} |")
    return "\n".join(out) + "\n"


def median_table(path: Path, title: str, intro: str, columns: list[tuple[str, str, int]], group: str | None = None) -> str:
    """A markdown table of medians (per `group` value, or per row) from a results CSV."""
    if not path.exists():
        return ""
    import pandas as pd
    df = pd.read_csv(path)
    if group:
        df = df.groupby(group, sort=False).median(numeric_only=True).reset_index()
    head = "| " + " | ".join(label for _, label, _ in columns) + " |"
    out = [f"## {title}", "", intro, "", head, "|" + "---|" * len(columns)]
    for _, r in df.iterrows():
        cells = []
        for col, _, d in columns:
            v = r[col]
            cells.append(str(v) if d < 0 else ("–" if pd.isna(v) else f"{v:.{d}f}"))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def comparison_tables() -> str:
    """AniMer and PRIMA against the fit, test-time refinement, body ablation and sizes (paper Tables main, refine, body, sizes)."""
    parts = [median_table(NEW / "baseline_animer" / "summary.csv", "Comparison with single-image networks",
                          "Medians per set, same detections and masks (`results/animer_baseline.csv`). table1 = the paper's per-clip "
                          "fitting clips; val/test = held-out snippets. LiDAR = held-out relief error in metres.",
                          [("set", "Set", -1), ("iou_animer", "IoU AniMer", 3), ("iou_prima", "IoU PRIMA", 3), ("iou_fit", "IoU WildPRIMA fit", 3),
                           ("lidar_held_animer", "LiDAR AniMer", 3), ("lidar_held_prima", "LiDAR PRIMA", 3), ("lidar_held_fit", "LiDAR WildPRIMA fit", 3)],
                          group="set")]
    refine = NEW / "eval_refine" / "summary.csv"
    if refine.exists():
        import pandas as pd
        r = pd.read_csv(refine)
        r[["w_orient", "w_relief"]] = r[["w_orient", "w_relief"]].fillna(0.0)
        r = r[(r.visits == 50) & (r.w_orient == 0) & r.w_relief.isin([0.0, 10.0])]
        tmp = NEW / "eval_refine" / "_card.csv"
        r.to_csv(tmp, index=False)
        parts.append(median_table(tmp, "Student-initialised refinement on video masks (no LiDAR)",
                                  "`python student_refine.py --split test --visits 50 --w-relief 10`: one body per track from the student, then "
                                  "pose refined on the masks with a pull of the depth relief towards the prediction; PRIMA refined identically "
                                  "for comparison (`results/refinement.csv`). Relief weight 10 was chosen on validation.",
                                  [("split", "Split", -1), ("w_relief", "Relief weight", 0), ("iou_prima_refined", "IoU PRIMA+ref.", 3),
                                   ("iou_student_refined", "IoU student+ref.", 3), ("lidar_held_prima_refined", "LiDAR PRIMA+ref.", 3),
                                   ("lidar_held_student_refined", "LiDAR student+ref.", 3), ("distance_rel_error_student_refined", "Dist. err", 3)],
                                  group=["split", "w_relief"]))
        tmp.unlink()
    parts.append(median_table(NEW / "ablation_body" / "summary_median.csv", "Per-clip fitting: body-model ablation",
                              "Medians over the seven clips (`results/body_ablation.csv` has every clip); `full` adds horns, humps and ears.",
                              [("variant", "Body model", -1), ("iou", "IoU", 3), ("top_recall", "Top recall", 3), ("held_err_m", "LiDAR (m)", 3)]))
    sizes = NEW / "sizes_summary.json"
    if sizes.exists():
        d = json.loads(sizes.read_text())
        rows = ["## Metric size against published adult heights", "",
                "Height at the withers, median (interquartile range) over all LiDAR-scaled tracks (`results/sizes_all_tracks.csv`).", "",
                "| Species | Tracks | Ours (m) | Published adult (m) |", "|---|---|---|---|"]
        for cls, v in d.items():
            lo, hi = v["published"]
            rows.append(f"| {cls} | {v['tracks']} | {v['median']:.2f} ({v['q1']:.2f}–{v['q3']:.2f}) | "
                        + (f"{lo:.2f}" if lo == hi else f"{lo:.2f}–{hi:.2f}") + " |")
        parts.append("\n".join(rows) + "\n")
    return "\n".join(p for p in parts if p)


def model_card(config: dict, summary: dict | None, prior_log: dict | None) -> str:
    test = (summary or {}).get("median_by_split", {}).get("test", {})
    f = lambda k, d=2: "–" if test.get(k) is None else f"{test[k]:.{d}f}"
    vae = "" if not prior_log else (f"Pose VAE: {prior_log['train']} training poses; mean joint error "
                                     f"{min(r['val_joint_error_deg_mean'] for r in prior_log['runs']):.1f}° on validation snippets.")
    return f"""---
license: other
license_name: prima-academic-non-commercial
license_link: LICENSE
library_name: pytorch
pipeline_tag: image-to-3d
tags:
- 3d-reconstruction
- animal-pose
- smal
- quadrupeds
- wildlife
- lidar
datasets:
- WildDepth
base_model: PRIMA
---

# WildPRIMA student ({config['version']})

![WildPRIMA teaser](teaser.jpg)
*{TEASER_CAPTION}*

Single-image, metric 3D reconstruction of quadrupeds. The model predicts SMAL pose and shape, 13 bone-group length and
thickness scales (an extended body that can express long necks and legs, unlike SMAL's shape space), and the animal's
metres per model unit, which with the camera intrinsics gives its distance and size in metres.

It is distilled from per-clip fits of WildDepth video with LiDAR (giraffe, Ankole-Watusi cattle, white rhinoceros,
Bactrian camel, lion, donkey), which use the extended body, a species prior for horns and humps, and a LiDAR relief term.

| | |
|---|---|
| Backbone | PRIMA ViT-H + BioCLIP token, **frozen, not included** (fetched from PRIMA) |
| Trained here | PRIMA's SMAL head (fine-tuned) + attention-pooled bone-scale / metric head |
| Parameters in `model.safetensors` | {config['parameters']['student']:,} |
| Pose prior (`pose_prior.safetensors`) | species-conditioned VAE, 6D rotations → {config['pose_prior']['latent']}-d latent, {config['parameters']['pose_prior']:,} parameters |
| Training | {config['training'].get('hours', '–')} h on one RTX PRO 4000 (24 GB), checkpoint at step {config['training'].get('step', '–')} |
| Data | {config['data'].get('label_windows', '–')} WildDepth windows; split by snippet (val/test snippets never seen) |

## Results (held-out test snippets, medians)

Silhouette IoU: PRIMA {f('iou_prima')}, per-clip fit {f('iou_fit')}, **student {f('iou_student')}**.
Held-out LiDAR relief error: PRIMA {f('lidar_held_prima', 3)} m, fit {f('lidar_held_fit', 3)} m, **student {f('lidar_held_student', 3)} m**.
Metric distance relative error: {f('distance_rel_error_student')}. Runtime: {f('student_ms_per_instance', 1)} ms per animal for the student
(backbone not counted).

The student's strength is metric output from one RGB image: distance and size without LiDAR at test time. In shape and pose
it is close to PRIMA, not yet to the per-clip fits it is trained on; when video (and ideally LiDAR) is available, the
per-clip fit (`run.py`) is the more accurate route. With video but no LiDAR, `student_refine.py` refines the student's
prediction on the masks; with the relief anchor chosen on validation (weight 10) it beats PRIMA refined the same way
on both silhouette and depth (tables below).

{results_table(summary)}
{vae}

{depth_ablation_table()}
{comparison_tables()}
## Use

```bash
git clone <github repo> wildprima && cd wildprima
bash setup_env.sh                      # conda env, PRIMA code + checkpoint, SMAL files (accept their licences)
conda activate prima
python predict.py photo.jpg --weights <hf user>/wildprima-student --calibration none --out predictions/
```

Outputs per animal: an OBJ in the camera frame in metres and a JSON with distance, metres per unit, body length and
height. Pass real intrinsics (`--calibration calib.json`) for metric use; with `none`, PRIMA's assumed focal length is used.

Loading the weights directly:

```python
from safetensors.torch import load_file
from student import Student, load_prima
prima = load_prima("cuda")
student = Student(prima.model.smal_head).cuda().eval()
student.load_state_dict(load_file("model.safetensors"))
out = student(tokens)  # tokens: (B, 193, 1280) from prima_tokens(); keys orient, pose, betas, cam, log_scale, log_mpu
```

## Limitations

- Horns and humps are not predicted by the student; the per-clip fit (`run.py`) adds them from the species table.
- Trained on six species at two sites (Longleat, UK; Bubye Valley, Zimbabwe), 10 Hz video. Other species are out of
  distribution; scale relies on LiDAR-supervised metres-per-unit and is least reliable for small or distant animals.
- Pseudo-labels come from optimisation on silhouettes and LiDAR, not ground-truth 3D; their errors are inherited.

## Licence

The weights are derived from PRIMA's checkpoint and are released under PRIMA's academic, non-commercial terms
(see `LICENSE`); they may not be used to harm animals. The code is MIT-licensed (GitHub). SMAL is licensed separately by MPI.

## Citation

```bibtex
@inproceedings{{wildprima2027,
  title     = {{WildPRIMA: Metric 3D Shape and Gait of Wild Quadrupeds from Video and LiDAR}},
  author    = {{Anonymous}},
  booktitle = {{CVPR}},
  year      = {{2027}}
}}
```
Please also cite PRIMA, SMAL and WildDepth (Aamir et al., ECCV 2026).
"""


def huggingface(out: Path, args) -> dict:
    from student import GROUPS, TOKENS
    from animal3d.prior import SPECIES_CLASSES

    out.mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(exist_ok=True)
    ckpt = torch.load(args.student, map_location="cpu")
    vae_ckpt = torch.load(args.prior, map_location="cpu")
    meta = {"format": FORMAT, "version": args.version}
    n_student = save_safetensors(ckpt["state"], out / "model.safetensors", meta)
    n_vae = save_safetensors(vae_ckpt["state"], out / "pose_prior.safetensors", meta)
    log_path = NEW / "student" / "train_log.jsonl"
    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()] if log_path.exists() else []
    summary = read_json(NEW / "eval" / "summary.json")
    prior_log = read_json(NEW / "prior" / "train_log.json")
    labels = read_json(NEW / "paper_numbers.json") or {}
    config = {
        "format": FORMAT, "name": f"wildprima-student-{args.version}", "version": args.version,
        "created": time.strftime("%Y-%m-%d"),
        "token_dim": 1280, "tokens": TOKENS, "dropout": 0.1, "bone_groups": GROUPS, "species_classes": list(SPECIES_CLASSES),
        "outputs": {"orient": "(B,3,3)", "pose": "(B,34,3,3)", "betas": "(B,41)", "cam": "(B,3) PRIMA crop camera",
                    "log_scale": "(B,13,2) log length/thickness per bone group", "log_mpu": "(B,) log metres per model unit"},
        "pose_prior": {"latent": 32, "rotation": "6d", "species_conditioned": True},
        "parameters": {"student": n_student, "pose_prior": n_vae},
        "requires": {"prima_checkpoint": "third_party/PRIMA/data/PRIMA/checkpoints/best_inference.ckpt (setup_env.sh)",
                     "smal": "PRIMA's SMAL files (MPI licence)"},
        "training": {"step": ckpt.get("step"), "val": {k: float(v) for k, v in (ckpt.get("val") or {}).items()},
                     "hours": round(records[-1]["hours"], 2) if records and "hours" in records[-1] else None},
        "data": {"dataset": "WildDepth (wilddepth_release)", "species": ["giraffe", "cattle", "rhino", "camel", "lion", "donkey"],
                 "label_windows": labels.get("labels", {}).get("windows_ok"), "usable_instances": labels.get("dataset", {}).get("usable"),
                 "split": read_json(NEW / "dataset" / "split.json")},
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    for src, dst in ((NEW / "eval" / "summary.json", "eval_summary.json"), (NEW / "eval" / "summary.csv", "eval_summary.csv"),
                     (log_path, "student_train_log.jsonl"), (NEW / "prior" / "train_log.json", "pose_prior_train_log.json"),
                     (NEW / "dataset" / "split.json", "split.json"),
                     (NEW / "ablation_depth" / "summary.csv", "depth_ablation.csv"),
                     (NEW / "ablation_body" / "summary.csv", "body_ablation.csv"),
                     (NEW / "baseline_animer" / "summary.csv", "animer_baseline.csv"),
                     (NEW / "eval_refine" / "summary.csv", "refinement.csv"),
                     (NEW / "sizes_all_tracks.csv", "sizes_all_tracks.csv"),
                     (NEW / "keypoint_clicks" / "summary.json", "keypoint_check.json"),
                     (NEW / "gait_all_tracks.csv", "gait_all_tracks.csv")):
        if src.exists():
            shutil.copy2(src, out / "results" / dst)
    shutil.copy2(TEASER, out / "teaser.jpg")
    (out / "README.md").write_text(model_card(config, summary, prior_log))
    shutil.copy2(ROOT / "third_party" / "PRIMA" / "LICENSE.md", out / "LICENSE")
    with open(out / "LICENSE", "a") as f:
        f.write("\n\n---\nThe WildPRIMA student weights in this repository are a derivative of the PRIMA model weights and are "
                "distributed under the terms above. The WildPRIMA code is MIT-licensed (see the GitHub repository).\n")
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text("".join(f"{sha256(p)}  {p.relative_to(out)}\n" for p in files))
    return config


def github(out: Path, config: dict) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "*.pdf")  # docs/ holds third-party papers
    for name in CODE:
        src = ROOT / name
        if src.is_dir():
            shutil.copytree(src, out / name, ignore=ignore)
        elif src.exists():
            (out / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out / name)
    shutil.copy2(TEASER, out / "assets" / "teaser.jpg")
    shutil.copy2(ROOT / "LICENSE", out / "LICENSE")
    (out / "NOTICE").write_text(
        "WildPRIMA code: MIT licence (LICENSE).\n"
        "Model weights (Hugging Face): derived from PRIMA, academic non-commercial use only (PRIMA licence).\n"
        "PRIMA (https://github.com/AdaptiveMotorControlLab/PRIMA) is cloned by setup_env.sh under its own licence.\n"
        "SMAL model files are licensed by the Max Planck Institute and are not redistributed here.\n"
        "Detectron2 Mask R-CNN weights are downloaded at run time under their own licence.\n")
    (out / ".gitignore").write_text("__pycache__/\n*.pyc\nthird_party/*\n!third_party/animer_loader.py\n!third_party/AniMer_SOURCE.txt\noutputs/\nnew_output/\nrelease/\npredictions/\n*.mp4\n")
    readme = (ROOT / "README.md").read_text()
    head = f"""# WildPRIMA: Metric 3D Shape and Gait of Wild Quadrupeds from Video and LiDAR

Metric 3D quadrupeds from wild video and single images: an extended SMAL body (bone scales, horns, humps) fitted per
clip with LiDAR, and a feed-forward student distilled from those fits (weights: Hugging Face, `{config['name']}`).

![WildPRIMA teaser](assets/teaser.jpg)
*{TEASER_CAPTION}*

```bash
bash setup_env.sh && conda activate prima
python predict.py photo.jpg --weights <hf user>/wildprima-student --out predictions/   # single image, student
python run.py <frames folder>                                                           # per-clip fitting
python student_refine.py --split test --w-relief 10                                     # student + refinement (eval)
python baseline_animer.py                                                               # AniMer baseline (needs third_party/AniMer)
```

Code: MIT. Weights: PRIMA's academic non-commercial licence. See NOTICE.

---

"""
    (out / "README.md").write_text(head + readme)
    req = (out / "requirements.txt").read_text()
    if "safetensors" not in req:
        (out / "requirements.txt").write_text(req.rstrip() + "\nsafetensors\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default="0.1.0")
    parser.add_argument("--student", type=Path, default=NEW / "student" / "student.pt")
    parser.add_argument("--prior", type=Path, default=NEW / "prior" / "vae.pt")
    parser.add_argument("--out", type=Path, default=ROOT / "release")
    args = parser.parse_args()
    if not args.student.exists():
        raise SystemExit(f"{args.student} not found: train the student first")
    config = huggingface(args.out / "huggingface", args)
    github(args.out / "github", config)
    print(f"release written to {args.out}: huggingface/ ({config['parameters']['student']:,} student parameters), github/")


if __name__ == "__main__":
    main()
