"""Fit labels for training: run the pipeline on windows of WildDepth release snippets, then extract kinematics.

    python make_labels.py [--release /mnt/hdd/WildDepth/wilddepth_release] [--out new_output/labels] [--hours 20]

Only the species in SPECIES are used. Long snippets are cut into up to MAX_WINDOWS evenly spaced windows of WINDOW
consecutive frames, because one shape fit per short window is cheaper than one per long snippet, and consecutive
frames are what gait needs. Each window becomes a folder of symlinks plus the snippet's metadata.txt, so its
calibration is found as usual. Finished windows are skipped, so the script can be stopped and restarted. It stops
starting new windows once `--hours` have passed.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPECIES = {"giraffe", "graffe", "ankle cow", "rhino", "camel", "lion", "lioness", "donkey"}
WINDOW, MAX_WINDOWS, FRAMES_PER_WINDOW = 300, 4, 1000  # one window per 1000 frames, up to 4
MAX_GAP = 3  # frame numbers: a longer jump starts a new stretch, since gait needs steady timing


def read_metadata(folder: Path) -> dict:
    path = folder / "metadata.txt"
    if not path.exists():
        return {}
    pairs = (line.split(":", 1) for line in path.read_text().splitlines() if ":" in line)
    return {k.strip().lower(): v.strip() for k, v in pairs}


def snippets(release: Path) -> list[Path]:
    found = []
    for meta in sorted(release.glob("*/**/metadata.txt")):
        if "extracted" in str(meta):
            continue
        folder = meta.parent
        if read_metadata(folder).get("species", "").lower() in SPECIES and (folder / "rgb").is_dir() and (folder / "depthm").is_dir():
            found.append(folder)
    return found


def windows(folder: Path) -> list[list[str]]:
    """Windows of up to WINDOW frames, each inside one stretch without gaps longer than MAX_GAP frame numbers."""
    depth = {p.stem for p in (folder / "depthm").iterdir()}
    frames = sorted((p.stem for p in (folder / "rgb").iterdir() if p.stem in depth), key=int)
    numbers = [int(f) for f in frames]
    stretches, start = [], 0
    for i in range(1, len(frames) + 1):
        if i == len(frames) or numbers[i] - numbers[i - 1] > MAX_GAP:
            stretches.append(frames[start:i])
            start = i
    count = max(1, min(MAX_WINDOWS, len(frames) // FRAMES_PER_WINDOW))
    # Spread the windows over the stretches long enough to hold one, longest first; short snippets give one window.
    long = sorted((s for s in stretches if len(s) >= WINDOW), key=len, reverse=True)
    if not long:
        return [max(stretches, key=len)]
    result = []
    for k in range(count):
        stretch = long[k % len(long)]
        per = (count + len(long) - 1 - (k % len(long))) // len(long)  # windows given to this stretch
        j = k // len(long)
        s = (len(stretch) - WINDOW) // 2 if per == 1 else round(j * (len(stretch) - WINDOW) / (per - 1))
        window = stretch[s:s + WINDOW]
        if window not in result:
            result.append(window)
    return result


def make_window(folder: Path, frames: list[str], target: Path) -> None:
    for sub in ("rgb", "depthm"):
        (target / sub).mkdir(parents=True, exist_ok=True)
        files = {p.stem: p for p in (folder / sub).iterdir()}
        for f in frames:
            link = target / sub / files[f].name
            if not link.exists():
                link.symlink_to(files[f])
    (target / "metadata.txt").write_text((folder / "metadata.txt").read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release", type=Path, default=Path("/mnt/hdd/WildDepth/wilddepth_release"))
    parser.add_argument("--out", type=Path, default=ROOT / "new_output" / "labels")
    parser.add_argument("--hours", type=float, default=20.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    jobs = []
    for folder in snippets(args.release):
        name = f"{folder.relative_to(args.release).parts[0]}_{folder.name}"
        for k, frames in enumerate(windows(folder)):
            jobs.append((folder, frames, f"{name}_w{k}"))
    print(f"{len(jobs)} windows from {len({j[0] for j in jobs})} snippets")
    if args.dry_run:
        for folder, frames, name in jobs:
            print(f"  {name}: {frames[0]}-{frames[-1]} ({len(frames)})")
        return

    inputs, runs = args.out / "inputs", args.out / "runs"
    index = args.out / "index.csv"
    args.out.mkdir(parents=True, exist_ok=True)
    new_index = not index.exists()
    started = time.time()
    with index.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if new_index:
            writer.writerow(["window", "snippet", "species", "first", "last", "frames", "status", "minutes"])
        for folder, frames, name in jobs:
            out = runs / name
            if (out / "summary.json").exists():
                continue
            if time.time() - started > args.hours * 3600:
                print(f"Time budget of {args.hours} h used; stopping before {name}")
                break
            make_window(folder, frames, inputs / name)
            t0 = time.time()
            log = (args.out / "logs").joinpath(f"{name}.log")
            log.parent.mkdir(exist_ok=True)
            with log.open("w") as stream:
                code = subprocess.call([sys.executable, str(ROOT / "run.py"), str(inputs / name), "--out", str(out), "--no-videos",
                                        "--mesh-stride", "0"], stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
            status = "ok" if code == 0 else f"exit {code}"
            if code == 0 and (out / "fit.npz").exists():
                code = subprocess.call([sys.executable, str(ROOT / "kinematics.py"), str(out)], stdout=log.open("a"),
                                       stderr=subprocess.STDOUT, cwd=ROOT)
                status = "ok" if code == 0 else f"kinematics exit {code}"
            elif code == 0:
                summary = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
                status = summary.get("result", "no fit")
            minutes = (time.time() - t0) / 60
            writer.writerow([name, str(folder), read_metadata(folder).get("species"), frames[0], frames[-1], len(frames), status, f"{minutes:.1f}"])
            handle.flush()
            print(f"{name}: {status}, {minutes:.1f} min", flush=True)


if __name__ == "__main__":
    main()
