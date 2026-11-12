"""Collect every result in new_output/ into numbers and LaTeX tables for the paper.

    python paper_results.py

Writes new_output/paper_numbers.json and paper/generated/{numbers,tab_student,tab_smoothing,tab_gait}.tex.
Missing parts (e.g. before the student has been evaluated) are skipped, so it can be rerun at any stage.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pose_prior import NEW, ROOT, run_folders, run_species, split_of
from animal3d.prior import species_class, SPECIES_CLASSES

GEN = ROOT / "paper" / "generated"
MIN_PERIODICITY, MIN_TRACK_FRAMES = 0.2, 30
CLASS_NAME = {"giraffe": "Giraffe", "cattle": "Cattle", "rhino": "Rhino", "camel": "Camel", "lion": "Lion",
              "donkey": "Donkey", "other": "Other"}


def fmt(x, digits=2):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{digits}f}"


def macro(name: str, value) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}\\xspace}}"


def labels(numbers: dict, macros: list[str]) -> None:
    index = pd.read_csv(NEW / "labels" / "index.csv")
    ok = index[index.status == "ok"]
    numbers["labels"] = {"windows": int(len(index)), "windows_ok": int(len(ok)), "frames": int(ok.frames.sum()),
                         "snippets": int(ok.snippet.nunique()), "gpu_hours": float(index.minutes.sum() / 60)}
    macros += [macro("NumWindows", len(ok)), macro("NumSnippets", ok.snippet.nunique()),
               macro("NumLabelFrames", f"{int(ok.frames.sum()):,}".replace(",", "{,}")),
               macro("LabelHours", fmt(index.minutes.sum() / 60, 1))]
    inst_file = NEW / "dataset" / "instances.csv"
    if inst_file.exists():
        inst = pd.read_csv(inst_file)
        usable = inst[inst.usable]
        numbers["dataset"] = {"instances": int(len(inst)), "usable": int(len(usable)),
                              "usable_by_split": usable.split.value_counts().to_dict(),
                              "usable_by_species": usable.species_class.map(lambda c: SPECIES_CLASSES[c]).value_counts().to_dict()}
        macros += [macro("NumInstances", f"{len(inst):,}".replace(",", "{,}")),
                   macro("NumUsable", f"{len(usable):,}".replace(",", "{,}"))]


def prior(numbers: dict, macros: list[str]) -> None:
    f = NEW / "prior" / "train_log.json"
    if not f.exists():
        return
    log = json.loads(f.read_text())
    best = min(log["runs"], key=lambda r: r["val_joint_error_deg_mean"])
    numbers["prior"] = {"train_poses": log["train"], "val_poses": log["val"], **best}
    macros += [macro("VAEValErr", fmt(best["val_joint_error_deg_mean"], 1)), macro("VAETrainErr", fmt(best["train_joint_error_deg_mean"], 1)),
               macro("VAETrainPoses", f"{log['train']:,}".replace(",", "{,}"))]


def smoothing(numbers: dict, macros: list[str]) -> None:
    f = NEW / "prior" / "smoothing" / "summary.csv"
    if not f.exists():
        return
    s = pd.read_csv(f)
    final = s[s.variant == "residual_final"].sort_values(["run", "w_smooth"])
    decode = s[(s.variant == "decode") & (s.w_smooth == 1e-3)]
    numbers["smoothing"] = s.to_dict(orient="records")
    rows = []
    for _, r in pd.concat([decode, final[final.w_smooth == 1e-2]]).iterrows():
        label = "decode" if r.variant == "decode" else "residual (ours)"
        rows.append(f"{r.run.split('_')[0].capitalize()} & {label} & {fmt(r.iou_fit)} / {fmt(r.iou_smoothed)} & "
                    f"{fmt(r.lidar_held_fit, 3)} / {fmt(r.lidar_held_smoothed, 3)} & {fmt(r.jitter_fit, 3)} / {fmt(r.jitter_smoothed, 3)} \\\\")
    (GEN / "tab_smoothing.tex").write_text(
        "\\begin{tabular}{llccc}\n\\toprule\nClip & Smoothing & IoU fit / smoothed & Held-out LiDAR (m) & Jitter \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def student(numbers: dict, macros: list[str]) -> None:
    f = NEW / "eval" / "summary.json"
    if not f.exists():
        return
    summary = json.loads(f.read_text())
    runs = pd.DataFrame(summary["runs"])
    numbers["student"] = summary
    log = NEW / "student" / "train_log.jsonl"
    if log.exists():
        records = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        numbers["student_training"] = {"records": len(records), "last": records[-1] if records else None}
    rows = []
    for split in ("val", "test"):
        part = runs[runs.split == split]
        if part.empty:
            continue
        rows.append(f"\\midrule\n\\multicolumn{{9}}{{l}}{{\\textit{{{split}}}}} \\\\")
        for _, r in part.iterrows():
            rows.append(f"\\texttt{{{r.run.replace('_', chr(92) + '_')}}} & {r.clean} & {fmt(r.iou_prima)} & {fmt(r.iou_fit)} & {fmt(r.iou_student)} & "
                        f"{fmt(r.lidar_held_prima, 3)} & {fmt(r.lidar_held_fit, 3)} & {fmt(r.lidar_held_student, 3)} & {fmt(r.distance_rel_error_student)} \\\\")
        med = part.median(numeric_only=True)
        rows.append(f"\\textbf{{median}} & {int(part.clean.sum())} & {fmt(med.iou_prima)} & {fmt(med.iou_fit)} & {fmt(med.iou_student)} & "
                    f"{fmt(med.lidar_held_prima, 3)} & {fmt(med.lidar_held_fit, 3)} & {fmt(med.lidar_held_student, 3)} & {fmt(med.distance_rel_error_student)} \\\\")
    (GEN / "tab_student.tex").write_text(
        "\\begin{tabular}{lrccccccc}\n\\toprule\n& & \\multicolumn{3}{c}{IoU $\\uparrow$} & \\multicolumn{3}{c}{Held-out LiDAR (m) $\\downarrow$} & Dist. \\\\\n"
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\nSnippet & $N$ & PRIMA & Fit & Student & PRIMA & Fit & Student & rel.\\,err \\\\\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    test = runs[runs.split == "test"].median(numeric_only=True)
    macros += [macro("StudentIoU", fmt(test.iou_student)), macro("PrimaIoU", fmt(test.iou_prima)), macro("FitIoU", fmt(test.iou_fit)),
               macro("StudentLidar", fmt(test.lidar_held_student, 3)), macro("PrimaLidar", fmt(test.lidar_held_prima, 3)),
               macro("FitLidar", fmt(test.lidar_held_fit, 3)), macro("StudentDistErr", fmt(100 * test.distance_rel_error_student, 0) + "\\%"),
               macro("StudentMs", fmt(runs.student_ms_per_instance.median(), 1))]


# Gait: tracks of at least MIN_TRACK_FRAMES frames that move (MIN_SPEED, from the LiDAR-scaled translation) with a clear
# stride peak (MIN_STRIDE_PERIODICITY of the left-right swing power). Existing clips that overlap a label window of the
# same snippet are left out so that no animal is counted twice.
MIN_SPEED, MIN_STRIDE_PERIODICITY = 0.3, 0.5
GAIT_OVERLAPS = {"giraffe_4_lidar", "Camel_1_domes"}  # overlap 19152060_giraffe_4_w0 and 19152060_camel_1_w0
# Basu, Wilson & Hutchinson 2019 (J Exp Biol 222, jeb159277): three adult giraffes, 0.74-1.3 m/s.
BASU = {"speed": (0.74, 1.3), "stride_duration": lambda u: -0.75 * u + 2.9, "stride_length": lambda u: 1.2 * u + 0.8,
        "relative_stride_length": 1.13, "limb_phase": 0.14, "duty_factor": 0.70, "neck_rom": lambda u: 6.4 * u + 3.1,
        "shoulder_height": (1.84, 1.87)}


def gait(numbers: dict, macros: list[str]) -> None:
    tracks = []
    for name, (out, _, _) in run_folders().items():
        if name in GAIT_OVERLAPS:
            continue
        f = out / "gait.json"
        if not f.exists():
            f = NEW / "kinematics_existing" / name / "gait.json"
        if not f.exists():
            continue
        cls = SPECIES_CLASSES[species_class(run_species(out))]
        for t in json.loads(f.read_text())["tracks"]:
            tracks.append({"run": name, "species": cls, **t})
    if not tracks:
        return
    df = pd.DataFrame(tracks)
    df["relative_stride_length"] = df.stride_length_m / df.shoulder_height_m
    df["froude"] = df.speed_m_s ** 2 / (9.81 * df.shoulder_height_m)
    df["walking"] = df.walking_bout & (df.frames >= MIN_TRACK_FRAMES) & (df.stride_periodicity >= MIN_STRIDE_PERIODICITY) & (df.speed_m_s >= MIN_SPEED)
    df.to_csv(NEW / "gait_all_tracks.csv", index=False)
    walking = df[df.walking]
    rows, joint_rows, stats = [], [], {}
    both = lambda part, c: np.nanmedian(np.r_[part[f"left_{c}"], part[f"right_{c}"]]) if len(part) else None
    med = lambda part, c: float(part[c].median()) if len(part) and part[c].notna().any() else None
    for cls in SPECIES_CLASSES:
        part, all_part = walking[walking.species == cls], df[(df.species == cls) & (df.frames >= MIN_TRACK_FRAMES)]
        if all_part.empty or cls == "other":  # not one of the six study species
            continue
        stats[cls] = {"tracks": int(len(all_part)), "walking_tracks": int(len(part)),
                      **{c: med(part, c) for c in ("stride_hz", "speed_m_s", "stride_length_m", "relative_stride_length",
                                                   "limb_phase", "duty_factor", "froude", "neck_range")},
                      **{f"{c}_range": both(part, f"{c}_range") for c in ("elbow", "carpus", "stifle", "hock", "fore", "hind")}}
        st = stats[cls]
        rows.append(f"{CLASS_NAME[cls]} & {st['walking_tracks']}/{st['tracks']} & {fmt(st['stride_hz'])} & {fmt(st['speed_m_s'])} & "
                    f"{fmt(st['stride_length_m'])} & {fmt(st['limb_phase'])} \\\\")
        joint_rows.append(f"{CLASS_NAME[cls]} & {st['walking_tracks']} & "
                          + " & ".join(fmt(st[f'{c}_range'], 0) for c in ("elbow", "carpus", "stifle", "hock")) + " \\\\")
    numbers["gait"] = {"min_speed": MIN_SPEED, "min_periodicity": MIN_STRIDE_PERIODICITY, "min_frames": MIN_TRACK_FRAMES,
                       "excluded_overlaps": sorted(GAIT_OVERLAPS), "species": stats}
    (GEN / "tab_gait.tex").write_text(
        "\\begin{tabular}{lccccc}\n\\toprule\n& Walking & Stride & Speed & Stride & Limb \\\\\n"
        "Species & tracks & (Hz) & (m/s) & length (m) & phase \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    # Joint ranges are not validated by the keypoint check (supplementary only).
    (GEN / "tab_gait_joints.tex").write_text(
        "\\begin{tabular}{lccccc}\n\\toprule\n& Walking & \\multicolumn{4}{c}{Joint range ($^\\circ$, 5--95\\%)} \\\\\n"
        "\\cmidrule(lr){3-6}\nSpecies & bouts & Elbow & Carpus & Stifle & Hock \\\\\n\\midrule\n"
        + "\n".join(joint_rows) + "\n\\bottomrule\n\\end{tabular}\n")
    giraffe_comparison(walking[walking.species == "giraffe"], numbers, macros)
    supplementary_tracks(walking)


def giraffe_comparison(g: pd.DataFrame, numbers: dict, macros: list[str]) -> None:
    """Walking giraffes against Basu et al. 2019, with their regressions evaluated at each track's own speed."""
    if g.empty:
        return
    u = g.speed_m_s.to_numpy()
    pred_hz = 1.0 / BASU["stride_duration"](u)
    pred_len = BASU["stride_length"](u)
    rng = lambda x, d=2: f"{fmt(float(np.nanmedian(x)), d)} ({fmt(float(np.nanmin(x)), d)}--{fmt(float(np.nanmax(x)), d)})"
    rows = [("Speed (m/s)", rng(u), "0.98 (0.74--1.30)"),
            ("Stride frequency (Hz)", rng(g.stride_hz), f"{fmt(float(np.median(pred_hz)))} at our speeds"),
            ("Stride length (m)", rng(g.stride_length_m), f"{fmt(float(np.median(pred_len)))} at our speeds"),
            ("Stride length / shoulder height", rng(g.relative_stride_length), "1.13"),
            ("Limb phase", rng(g.limb_phase), "0.14"),
            ("Duty factor", rng(g.duty_factor), "0.70"),
            ("Neck range ($^\\circ$)", rng(g.neck_range, 0), f"{fmt(float(np.median(BASU['neck_rom'](u))), 0)} at our speeds"),
            ("Shoulder height (m)", rng(g.shoulder_height_m), "1.84--1.87")]
    err_hz = np.abs(g.stride_hz.to_numpy() - pred_hz) / pred_hz
    err_len = np.abs(g.stride_length_m.to_numpy() - pred_len) / pred_len
    numbers["gait_vs_basu"] = {"tracks": int(len(g)), "runs": sorted(g.run.unique()),
                               "stride_hz_rel_err_median": float(np.median(err_hz)), "stride_length_rel_err_median": float(np.median(err_len)),
                               "rows": [{"quantity": a, "ours": b, "basu": c} for a, b, c in rows]}
    (GEN / "tab_basu.tex").write_text(
        f"\\begin{{tabular}}{{lcc}}\n\\toprule\n& Ours, {len(g)} tracks & Basu \\etal \\\\\n\\midrule\n"
        + "\n".join(f"{a} & {b} & {c} \\\\" for a, b, c in rows) + "\n\\bottomrule\n\\end{tabular}\n")
    macros += [macro("GiraffeWalkTracks", len(g)), macro("BasuHzErr", fmt(100 * float(np.median(err_hz)), 0) + "\\%"),
               macro("BasuLenErr", fmt(100 * float(np.median(err_len)), 0) + "\\%")]


def supplementary_tracks(walking: pd.DataFrame) -> None:
    """Every walking track, for the supplementary material."""
    rows = []
    for _, r in walking.sort_values(["species", "run", "track_id"]).iterrows():
        rows.append(f"{CLASS_NAME[r.species]} & \\texttt{{{r.run.replace('_', chr(92) + '_')}}} & {r.track_id} & {r.frames} & {fmt(r.stride_hz)} & "
                    f"{fmt(r.speed_m_s)} & {fmt(r.stride_length_m)} & {fmt(r.relative_stride_length)} & {fmt(r.limb_phase)} & "
                    f"{fmt(r.duty_factor)} & {fmt(r.neck_range, 0)} \\\\")
    (GEN / "tab_gait_tracks.tex").write_text(
        "\\begin{tabular}{llrrcccccccc}\n\\toprule\nSpecies & Window & Track & Frames & Stride (Hz) & Speed (m/s) & Stride (m) & "
        "Stride/height & Limb phase & Duty factor & Neck ($^\\circ$) \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def main() -> None:
    GEN.mkdir(parents=True, exist_ok=True)
    numbers, macros = {}, ["% generated by paper_results.py; do not edit"]
    for part in (labels, prior, smoothing, student, gait):
        part(numbers, macros)
    (GEN / "numbers.tex").write_text("\n".join(macros) + "\n")
    (NEW / "paper_numbers.json").write_text(json.dumps(numbers, indent=2, default=float))
    print(f"wrote {GEN} ({len(macros) - 1} macros) and {NEW / 'paper_numbers.json'}")


if __name__ == "__main__":
    main()
