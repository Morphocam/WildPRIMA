#!/usr/bin/env python
"""A small web GUI for the pipeline: choose an image, a folder or a video, then see the 3D result.

    python app.py                 # http://127.0.0.1:7860
    python app.py --host 0.0.0.0  # reachable from other machines; anyone who can reach it can run jobs

Each job runs `run.py` as a separate process, so the GUI stays responsive and Stop can end it cleanly.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from animal3d.sources import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
ASSETS = ROOT / "assets"
UPLOADS = OUTPUTS / "_uploads"
# Species in WildDepth or asked for by name. It only labels the outputs: the fit never sees it.
SPECIES = ["(from metadata, else unknown)", "giraffe", "horse", "donkey", "cow", "ankole cow", "steenbok", "goat", "rhino",
           "lion", "tiger", "wildebeest", "zebra", "camel", "cheetah", "wolf", "oryx", "springbok", "gemsbok", "warthog"]
LOG_LINES = 40

_job_lock = threading.Lock()
_job: subprocess.Popen | None = None


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._") or "input"


def _is_depth_png(path: Path) -> bool:
    if path.suffix.lower() != ".png":
        return False
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return image is not None and image.dtype == np.uint16 and image.ndim == 2


def arrange_upload(files: list[str], target: Path) -> Path:
    """Rebuild an uploaded folder on disk, in the layout `run.py` expects.

    Browsers flatten the folder, so rgb/ and depthm/ arrive mixed together. A 16-bit single-channel PNG is
    a depth map, and is only paired up if a colour image has the same name.
    """
    target.mkdir(parents=True, exist_ok=True)
    paths = [Path(f) for f in files]
    depth = [p for p in paths if _is_depth_png(p)]
    images = [p for p in paths if p.suffix.lower() in IMAGE_EXTENSIONS and p not in depth]
    metadata = [p for p in paths if p.name == "metadata.txt"]
    if not images:
        raise gr.Error("That folder has no images.")
    image_stems = {p.stem for p in images}
    depth = [p for p in depth if p.stem in image_stems]
    if depth:
        (target / "rgb").mkdir(exist_ok=True)
        (target / "depthm").mkdir(exist_ok=True)
    for path in images:
        shutil.copy(path, (target / "rgb" if depth else target) / path.name)
    for path in depth:
        shutil.copy(path, target / "depthm" / path.name)
    for path in metadata[:1]:
        shutil.copy(path, target / "metadata.txt")
    return target


def resolve_input(kind: str, image, folder, video, server_path: str) -> tuple[Path, str]:
    """The path to give run.py, and a name for its results."""
    if kind == "Image":
        if not image:
            raise gr.Error("Choose an image first.")
        path = Path(image)
        return path, safe_name(path.stem)
    if kind == "Video":
        if not video:
            raise gr.Error("Choose a video first.")
        path = Path(video)
        return path, safe_name(path.stem)
    if kind == "Folder of images":
        if not folder:
            raise gr.Error("Choose a folder first.")
        files = [str(f) for f in folder]
        return arrange_upload(files, UPLOADS / time.strftime("%Y%m%d_%H%M%S")), "folder"
    if not server_path or not server_path.strip():
        raise gr.Error("Enter a path on this machine.")
    path = Path(server_path.strip()).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise gr.Error(f"Not found: {path}")
    return path, safe_name(path.stem if path.is_file() else path.name)


def build_command(path: Path, out: Path, species: str, threshold: float, max_frames: float, quick: bool) -> list[str]:
    command = [sys.executable, "-u", str(ROOT / "run.py"), str(path), "--out", str(out), "--det-thresh", f"{threshold:g}"]
    if species and species != SPECIES[0]:
        command += ["--species", species]
    if max_frames:
        command += ["--max-frames", str(int(max_frames))]
    if quick:
        command.append("--baseline-only")
    return command


def format_results(out: Path) -> str:
    """A short Markdown summary of what the run found, from summary.json."""
    path = out / "summary.json"
    if not path.exists():
        return "No summary was written; see the log."
    s = json.loads(path.read_text())
    if s.get("result") == "no animals detected":
        return "### No animals detected\nTry a lower detection threshold."
    lines = [f"### {s['input']['species']}: {s['instances']} animal instances in {s['run']['frames_processed']} frames, "
             f"{s['tracks']} tracks", f"Results are in `{out}`."]
    clean = s.get("clean_all")
    if clean:
        shape = s["shape"]
        lines += ["", "| Median over clean instances | PRIMA | Fitted body |", "|---|---:|---:|",
                  f"| Silhouette IoU with mask | {clean['iou']['prima']:.2f} | {clean['iou']['fitted']:.2f} |",
                  f"| Neck / body length | {shape['proportions']['prima']['neck_to_body']:.2f} | {shape['proportions']['fitted']['neck_to_body']:.2f} |",
                  f"| Leg / body length | {shape['proportions']['prima']['leg_to_body']:.2f} | {shape['proportions']['fitted']['leg_to_body']:.2f} |"]
        lidar = s.get("lidar")
        if lidar:
            lines.append(f"| LiDAR surface error (m) | {lidar['relief_error_m']['prima']:.2f} | {lidar['relief_error_m']['fitted']:.2f} |")
        size = s.get("model_size", {})
        if lidar and lidar.get("tracks"):
            lines += ["", f"**Measured with LiDAR:** shoulder height {size['shoulder_height']:.2f} m, "
                          f"body and head length {size['length']:.2f} m. No ground truth was available to check these."]
        elif lidar:
            lines += ["", "Too few LiDAR views of any one animal to give sizes in metres."]
        else:
            lines += ["", "No LiDAR with a known calibration, so sizes in metres are not available."]
        if shape["measured_on_instances"] < 20:
            lines += ["", f"**Low confidence:** the body was measured on only {shape['measured_on_instances']} views. "
                          "One viewpoint cannot determine a body's depth or thickness."]
    elif "clean" in s:
        lines += ["", f"PRIMA only. Median silhouette IoU with the mask: {s['clean']['iou']:.2f}."]
    return "\n".join(lines)


def model_viewer_html(glb: str | None) -> str | None:
    """A 3D viewer for a GLB file, in an iframe because scripts do not run directly in gr.HTML.

    Gradio's own Model3D needs WebGPU and shows nothing without it, as in Chrome on many Linux setups.
    <model-viewer> (@google/model-viewer 3.5.0, Apache-2.0, in assets/) uses WebGL and works offline.
    """
    if glb is None:
        return None
    script, model = f"/gradio_api/file={ASSETS / 'model-viewer.min.js'}", f"/gradio_api/file={glb}"
    page = (f'<script type="module" src="{script}"></script><body style="margin:0;background:#f7f7f5">'
            f'<model-viewer src="{model}" camera-controls interaction-prompt="none" shadow-intensity="0.6" exposure="1.1" '
            f'style="width:100%;height:100vh" alt="3D model of the animal"></model-viewer></body>')
    return f"<iframe title=\"3D model\" srcdoc='{page}' style=\"width:100%;height:420px;border:0\"></iframe>"


def find_outputs(out: Path) -> dict:
    existing = lambda *names: next((str(out / n) for n in names if (out / n).exists()), None)
    stills = sorted((out / "overlays").glob("*_comparison.jpg")) or sorted((out / "overlays").glob("*_overlay.jpg")) if (out / "overlays").is_dir() else []
    downloads = [str(out / n) for n in ("model.glb", "model.obj", "summary.json", "metrics.csv", "comparison.mp4", "overlay.mp4", "model_turntable.mp4") if (out / n).exists()]
    return {
        "video": existing("comparison.mp4", "overlay.mp4"),
        "stills": [str(p) for p in stills[:40]] or None,
        "model": model_viewer_html(existing("model.glb")),
        "turntable": existing("model_turntable.mp4"),
        "downloads": downloads or None,
    }


def run_job(kind, image, folder, video, server_path, species, threshold, max_frames, quick):
    """Runs the pipeline, updating the log as it goes. Outputs: status, log, then the result views."""
    global _job
    empty = tuple(gr.update(value=None, visible=False) for _ in range(5))
    if not _job_lock.acquire(blocking=False):
        raise gr.Error("A job is already running. Stop it or wait for it to finish.")
    try:
        path, name = resolve_input(kind, image, folder, video, server_path)
        # A new folder per run: a quick run must never show files left by an earlier, fuller one.
        out = OUTPUTS / "gui" / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
        out.mkdir(parents=True, exist_ok=True)
        command = build_command(path, out, species, threshold, max_frames, quick)
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "TQDM_MININTERVAL": "1"}
        # Own process group, so that Stop reaches everything the pipeline started.
        _job = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)

        lines: list[str] = []
        buffer = b""
        last_update = 0.0
        yield ("Running", "", "", *empty)
        while True:
            chunk = _job.stdout.read1(4096)
            if not chunk:
                break
            buffer += chunk
            # tqdm rewrites its line with \r: show only the newest version of it.
            *complete, buffer = re.split(rb"[\r\n]", buffer)
            for line in complete:
                text = line.decode(errors="replace").strip()
                # run.py also prints its summary as JSON, which format_results shows far more readably.
                if not text or "Warning" in text or "warnings.warn" in text or re.match(r'^[{}\[\]"]', text):
                    continue
                if lines and re.match(r"^[\w +]+:\s+\d+%|^[\w +]+: \d+", text) and text.split(":")[0] == lines[-1].split(":")[0]:
                    lines[-1] = text
                else:
                    lines.append(text)
            lines = lines[-LOG_LINES:]
            if time.time() - last_update > 0.5:
                last_update = time.time()
                yield ("Running", "\n".join(lines), "", *empty)

        code = _job.wait()
        log = "\n".join(lines)
        (out / "gui_run.log").write_text(log)
        if code in (-signal.SIGTERM, -signal.SIGKILL):
            yield ("Stopped", log, "Stopped before finishing.", *empty)
            return
        if code != 0:
            yield (f"Failed (exit code {code})", log, "The pipeline failed; the log above shows why.", *empty)
            return
        found = find_outputs(out)
        views = [gr.update(value=found[k], visible=bool(found[k])) for k in ("video", "stills", "model", "turntable", "downloads")]
        yield ("Finished", log, format_results(out), *views)
    finally:
        _job = None
        _job_lock.release()


def stop_job() -> str:
    if _job is not None and _job.poll() is None:
        os.killpg(_job.pid, signal.SIGTERM)
        return "Stopping"
    return "Nothing is running"


def show_input(kind: str):
    return (gr.update(visible=kind == "Image"), gr.update(visible=kind == "Folder of images"),
            gr.update(visible=kind == "Video"), gr.update(visible=kind == "Path on this machine"))


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="WildPRIMA") as demo:
        gr.Markdown("# WildPRIMA\nBuild a 3D model of the animals in an image, a folder of images or a video.")
        with gr.Row():
            with gr.Column(scale=1):
                kind = gr.Radio(["Image", "Folder of images", "Video", "Path on this machine"], value="Image", label="Input")
                image = gr.File(label="Image", file_count="single", file_types=sorted(IMAGE_EXTENSIONS), type="filepath")
                folder = gr.File(label="Folder of images (one species)", file_count="directory", type="filepath", visible=False)
                video = gr.File(label="Video", file_count="single", file_types=sorted(VIDEO_EXTENSIONS), type="filepath", visible=False)
                server_path = gr.Textbox(label="Path", placeholder="input_30s/giraffe_4", visible=False,
                                         info="Faster for large folders, as nothing is uploaded. An image, video or folder, relative to the project folder.")
                species = gr.Dropdown(SPECIES, value=SPECIES[0], allow_custom_value=True, label="Species", info="Chooses which horns to try (paired, on the nose, or none); recognised from the images if left unset. The body itself is measured from the images.")
                with gr.Accordion("Options", open=False):
                    threshold = gr.Slider(0.3, 0.95, value=0.7, step=0.05, label="Detection threshold", info="Lower it if animals are missed; too low admits partial bodies.")
                    max_frames = gr.Number(value=0, precision=0, minimum=0, label="Maximum frames (0 = all)")
                    quick = gr.Checkbox(value=False, label="PRIMA only (fast, but keeps PRIMA's generic body)")
                with gr.Row():
                    start = gr.Button("Run", variant="primary")
                    stop = gr.Button("Stop")
                status = gr.Textbox(label="Status", value="Ready", interactive=False)
                log = gr.Textbox(label="Log", lines=12, max_lines=12, interactive=False, autoscroll=True)
            with gr.Column(scale=2):
                results = gr.Markdown()
                result_video = gr.Video(label="PRIMA compared with the fitted body", interactive=False, visible=False)
                stills = gr.Gallery(label="Results for each image", columns=2, height="auto", visible=False)
                with gr.Row():
                    model = gr.HTML(label="3D model (drag to rotate)", show_label=True, visible=False)
                    turntable = gr.Video(label="Model turntable", interactive=False, visible=False)
                downloads = gr.File(label="Downloads", file_count="multiple", interactive=False, visible=False)

        kind.change(show_input, kind, [image, folder, video, server_path])
        start.click(run_job, [kind, image, folder, video, server_path, species, threshold, max_frames, quick],
                    [status, log, results, result_video, stills, model, turntable, downloads], concurrency_limit=1)
        stop.click(stop_job, None, status)
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    OUTPUTS.mkdir(exist_ok=True)
    build_ui().queue().launch(server_name=args.host, server_port=args.port, allowed_paths=[str(OUTPUTS), str(ASSETS)], inbrowser=False)


if __name__ == "__main__":
    main()
