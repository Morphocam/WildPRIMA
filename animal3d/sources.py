"""Reads an image, a folder of images, a video, or an RGB + LiDAR snippet as one stream of frames.

Depth units and intrinsics differ between WildDepth sites, and a wrong unit silently scales every
measurement by 10x, so depth is only used with a known calibration. Without one, results stay unitless.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


@dataclass(frozen=True)
class Camera:
    """Pinhole intrinsics in pixels."""

    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class Calibration:
    """A sensor rig: intrinsics for a given image size, and metres per unit of the depth PNGs."""

    name: str
    camera: Camera
    image_size: tuple[int, int]  # (width, height) the intrinsics apply to
    metres_per_unit: float


# Values from each record's calibration/intrinsic_1080p_rectified.txt. Rectification left fx != fy, so
# these frames have non-square pixels. Units: Bubye's README says centimetres. Longleat's records have no
# README; its values top out near 49,000 under the rig's 50 m cutoff, so millimetres.
CALIBRATIONS = {
    "longleat_v2": Calibration("longleat_v2", Camera(902.7877, 1069.3145, 844.8049, 548.9076), (1920, 1080), 0.001),
    "bubye": Calibration("bubye", Camera(902.5254, 1069.3145, 844.5594, 548.9076), (1920, 1080), 0.01),
}


def natural_key(path: Path) -> list:
    """Sorts 'f2.jpg' before 'f10.jpg'."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.stem)]


def read_metadata(path: Path) -> dict[str, str]:
    """Parses a WildDepth metadata.txt of 'key: value' lines; empty if it does not exist."""
    if not path.is_file():
        return {}
    pairs = (line.split(":", 1) for line in path.read_text(errors="replace").splitlines() if ":" in line)
    return {key.strip().lower(): value.strip() for key, value in pairs}


def calibration_from_metadata(metadata: dict[str, str]) -> Calibration | None:
    source = metadata.get("source", "").lower()
    if source.startswith("zimbabwe"):
        return CALIBRATIONS["bubye"]
    if source.startswith("july"):  # Longleat recording days (July07, July08)
        return CALIBRATIONS["longleat_v2"]
    return None


def load_calibration(spec: str, folder: Path | None, metadata: dict[str, str]) -> Calibration | None:
    """`spec`: 'auto', 'none', a name in CALIBRATIONS, or a JSON file.

    A JSON file holds {"fx", "fy", "cx", "cy", "width", "height", "metres_per_unit"}. 'auto' uses
    calibration.json in the input folder if there is one, otherwise the site named by metadata.txt.
    """
    if spec == "none":
        return None
    if spec in CALIBRATIONS:
        return CALIBRATIONS[spec]
    path = Path(spec) if spec != "auto" else (folder / "calibration.json" if folder else None)
    if path is not None and path.is_file():
        raw = json.loads(path.read_text())
        camera = Camera(float(raw["fx"]), float(raw["fy"]), float(raw["cx"]), float(raw["cy"]))
        return Calibration(path.name, camera, (int(raw["width"]), int(raw["height"])), float(raw["metres_per_unit"]))
    if spec != "auto":
        raise SystemExit(f"Unknown calibration: {spec}. Use auto, none, {', '.join(CALIBRATIONS)} or a JSON file.")
    return calibration_from_metadata(metadata)


def scaled_camera(calibration: Calibration, width: int, height: int) -> Camera:
    """The calibration's intrinsics for frames resized to (width, height)."""
    sx, sy = width / calibration.image_size[0], height / calibration.image_size[1]
    c = calibration.camera
    return Camera(c.fx * sx, c.fy * sy, c.cx * sx, c.cy * sy)


def move_to_camera(cam_t: np.ndarray, focal: float, width: int, height: int, camera: Camera) -> np.ndarray:
    """Translation (N, 3) from PRIMA's assumed camera to `camera`, keeping the animal where it is in the image.

    PRIMA assumes a long focal length and a centred principal point. A real lens puts the animal much
    closer in model units, and depth follows focal length so that its size in pixels stays about the same.
    """
    cam_t = np.atleast_2d(cam_t)
    depth = cam_t[:, 2] * (camera.fx + camera.fy) / 2 / focal
    u = focal * cam_t[:, 0] / cam_t[:, 2] + width / 2
    v = focal * cam_t[:, 1] / cam_t[:, 2] + height / 2
    return np.stack([depth * (u - camera.cx) / camera.fx, depth * (v - camera.cy) / camera.fy, depth], axis=1)


@dataclass
class Frame:
    index: int
    name: str
    bgr: np.ndarray
    depth_m: np.ndarray | None  # same size as bgr; 0 where the LiDAR has no point


class FrameSource:
    """Iterates frames in order. `sequence` is True when consecutive frames show the same scene over time."""

    def __init__(self, name: str, kind: str, fps: float, species: str, calibration: Calibration | None,
                 video: Path | None = None, images: list[Path] | None = None, depths: dict[str, Path] | None = None):
        self.name, self.kind, self.fps, self.species, self.calibration = name, kind, fps, species, calibration
        self.video, self.images, self.depths = video, images or [], depths or {}

    @property
    def sequence(self) -> bool:
        return self.kind in ("video", "sequence")

    @property
    def has_lidar(self) -> bool:
        return bool(self.depths) and self.calibration is not None

    def __iter__(self) -> Iterator[Frame]:
        if self.video is not None:
            cap = cv2.VideoCapture(str(self.video))
            index = 0
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                yield Frame(index, f"{index:05d}", bgr, None)
                index += 1
            cap.release()
            return
        for index, path in enumerate(self.images):
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise SystemExit(f"Cannot read image: {path}")
            yield Frame(index, path.stem, bgr, self._depth(path.stem, bgr.shape[:2]))

    def _depth(self, stem: str, shape: tuple[int, int]) -> np.ndarray | None:
        if not self.has_lidar or stem not in self.depths:
            return None
        raw = cv2.imread(str(self.depths[stem]), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.dtype != np.uint16 or raw.shape != shape:
            raise SystemExit(f"Expected a 16-bit depth PNG of size {shape}: {self.depths[stem]}")
        return raw.astype(np.float32) * self.calibration.metres_per_unit

    def describe(self) -> dict:
        calibration = None
        if self.calibration is not None:
            calibration = {**asdict(self.calibration.camera), "name": self.calibration.name,
                           "metres_per_unit": self.calibration.metres_per_unit}
        return {"name": self.name, "kind": self.kind, "species": self.species, "lidar": self.has_lidar, "calibration": calibration}


def _images_in(folder: Path) -> list[Path]:
    return sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS), key=natural_key)


def open_source(path: Path, species: str | None = None, calibration: str = "auto") -> FrameSource:
    """Works out what `path` is. `species` overrides any species in metadata.txt; it is only a label."""
    path = Path(path).resolve()
    if not path.exists():
        raise SystemExit(f"Input not found: {path}")

    if path.is_file():
        suffix = path.suffix.lower()
        if suffix in VIDEO_EXTENSIONS:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise SystemExit(f"Cannot open video: {path}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
            return FrameSource(path.stem, "video", fps, species or "animal", load_calibration(calibration, None, {}), video=path)
        if suffix in IMAGE_EXTENSIONS:
            return FrameSource(path.stem, "image", 1.0, species or "animal", load_calibration(calibration, None, {}), images=[path])
        raise SystemExit(f"Unsupported file type {suffix}. Images: {sorted(IMAGE_EXTENSIONS)}; videos: {sorted(VIDEO_EXTENSIONS)}")

    # A snippet folder (<folder>/rgb + depthm + metadata.txt); `rgb` itself also finds its parent's data.
    root = path.parent if path.name == "rgb" and (path.parent / "depthm").is_dir() else path
    rgb_dir = root / "rgb" if (root / "rgb").is_dir() else path
    images = _images_in(rgb_dir)
    if not images:
        children = sorted(p.name for p in path.iterdir() if p.is_dir())
        hint = f" It has subfolders ({', '.join(children[:8])}); choose one of those." if children else ""
        raise SystemExit(f"No images in {path}.{hint}")

    metadata = read_metadata(root / "metadata.txt")
    label = species or metadata.get("species") or "animal"
    cal = load_calibration(calibration, root, metadata)
    if (root / "depthm").is_dir():
        depths = {p.stem: p for p in (root / "depthm").glob("*.png")}
        return FrameSource(root.name, "sequence", _FPS_SNIPPET, label, cal, images=images, depths=depths)
    return FrameSource(root.name, "images", 1.0, label, cal, images=images)


_FPS_SNIPPET = 10.0  # WildDepth snippets are recorded at 10 Hz
