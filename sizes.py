"""Item 6: fitted metric sizes against published adult measurements (paper Table tab:sizes).

    python sizes.py

Our `shoulder_height` (run.py) is the height of SMAL's spine joint at the shoulders in the clip's mean pose, inside the
body. Published heights are taken at the withers, on the back above that joint, so we add the body's thickness there
(highest rest-pose vertex on the dorsal midline above joint 6, in model units, times the track's LiDAR metres per unit).
Writes new_output/sizes_all_tracks.csv and paper/generated/tab_sizes.tex.
"""
import json

import numpy as np
import pandas as pd
import torch

from animal3d.body import BODY_BOUNDS, AnimalBody
from animal3d.prior import SPECIES_CLASSES, species_class
from pose_prior import NEW, run_folders, run_species
from run import ROOT, load_smal

# Published adult height at the withers (m): (low, high) range or mean, and the bib key.
PUBLISHED = {
    "giraffe": ((2.72, 3.47), "female 2.72--2.92, male 3.13--3.47", "sdzwa_giraffe"),
    "cattle": ((1.29, 1.29), "mean 1.29 (Ankole)", "petersen2004ankole"),
    "rhino": ((1.70, 1.85), "1.70--1.85 (white)", "sdzwa_rhino"),
    "camel": ((1.60, 1.80), "1.60--1.80 (Bactrian)", "sdzwa_camel"),
    "lion": ((1.07, 1.23), "female 1.07, male 1.23 (means)", "sdzwa_lion"),
    "donkey": ((1.05, 1.05), "mean 1.05, range 0.91--1.20", "nengomasha1998donkey"),
}
NAME = {"giraffe": "Giraffe", "cattle": "Cattle", "rhino": "Rhinoceros", "camel": "Camel", "lion": "Lion", "donkey": "Donkey"}


def main() -> None:
    smal = load_smal(ROOT / "third_party/PRIMA")
    rows = []
    for name, (out, _, _) in run_folders().items():
        if not (out / "summary.json").exists() or not (out / "fit.npz").exists():
            continue
        summary, fit = json.loads((out / "summary.json").read_text()), np.load(out / "fit.npz")
        body = AnimalBody(**smal, betas=fit["betas"], bounds=BODY_BOUNDS)
        with torch.no_grad():
            body.log_scale.copy_(torch.as_tensor(fit["log_scale"]))
            joints = body.rest()[1].numpy()
        v = fit["rest_vertices"][:3889]  # without horns and humps
        above = (np.abs(v[:, 0] - joints[6, 0]) < 0.04) & (np.abs(v[:, 1] - joints[6, 1]) < 0.06)
        thickness = float(v[above, 2].max() - joints[6, 2])
        cls = SPECIES_CLASSES[species_class(run_species(out))]
        for t in summary.get("lidar", {}).get("tracks", []):
            if t.get("shoulder_height_m") is None:
                continue
            rows.append({"run": name, "species": cls, "track": t["track_id"], "instances": t["instances"],
                         "scale_spread": t.get("scale_spread"), "metres_per_unit": t["metres_per_unit"],
                         "joint_height_m": t["shoulder_height_m"], "withers_m": t["shoulder_height_m"] + thickness * t["metres_per_unit"],
                         "top_height_m": t.get("top_height_m"), "length_m": t.get("length_m"), "depth_m": t.get("depth_median_m")})
    df = pd.DataFrame(rows)
    df.to_csv(NEW / "sizes_all_tracks.csv", index=False)
    lines = [r"\begin{tabular}{@{}lrccl@{}}", r"\toprule",
             r"Species & Tracks & Ours (m) & Published adult (m) & $\Delta$ \\", r"\midrule"]
    report = {}
    for cls, ((lo, hi), text, key) in PUBLISHED.items():
        g = df[df.species == cls].withers_m
        med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
        delta = 0.0 if lo <= med <= hi else (med - lo) / lo if med < lo else (med - hi) / hi
        report[cls] = {"tracks": len(g), "median": med, "q1": q1, "q3": q3, "published": [lo, hi], "delta": delta}
        lines.append(f"{NAME[cls]} & {len(g)} & {med:.2f} ({q1:.2f}--{q3:.2f}) & {text} \\cite{{{key}}} & "
                     + ("in range" if delta == 0 else f"{100 * delta:+.0f}\\%") + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper" / "generated" / "tab_sizes.tex").write_text("% made by sizes.py\n" + "\n".join(lines) + "\n")
    (NEW / "sizes_summary.json").write_text(json.dumps(report, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
