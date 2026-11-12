"""Joint angles and gait from fitted bodies: the skeleton's angles per frame, and stride timing per animal.

SMAL's leg joints sit roughly where the anatomical ones are, but they are not anatomical landmarks: there is no
scapula, the "shoulder" joint is where the leg leaves the trunk, and the foot ends at the fetlock. Angles here are
the skeleton's, read from the fit, so they are only as good as the fitted pose. The joint names used:

    fore: shoulder 7 -> elbow 8 -> carpus 9 -> fetlock 10      hind: hip 17 -> stifle 18 -> hock 19 -> fetlock 20
    (left; the right legs are 11-14 and 21-24)

Inner angle: 180 degrees is a straight limb. Flexion is 180 minus that. Limb angle: the angle of the line from the
top of the leg to the fetlock against the body's vertical, in the body's side plane, positive forwards. A walking
animal swings it back and forth once per stride, which is what the stride timing is read from.

Neck angle: the angle of the neck bone (joint 15 -> head joint 16) above the body's forward axis, in the side plane.

Stride timing uses left minus right swing of each limb pair. In every symmetric gait (walk, pace, trot) the two legs of
a pair move half a stride apart, so the difference oscillates once per stride. An earlier version used fore minus hind on
one side, which cancels in a lateral-sequence walk (giraffes: limb phase 0.14, Basu et al. 2019) and doubles the frequency.
Limb phase and duty factor follow Basu et al.: limb phase is the fraction of the stride by which a forelimb's footfall
follows the ipsilateral hindlimb's (peak protraction is taken as footfall); duty factor is the fraction of the stride
a foot is on the ground, estimated as the fraction of time the limb rotates backwards relative to the body (retraction),
on the swing signal smoothed to three harmonics of the stride. At 10 Hz both are approximate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from smplx.lbs import batch_rigid_transform

# (name, parent joint, joint, child joint): the angle is measured at the middle joint.
JOINTS = {
    "left_elbow": (7, 8, 9), "left_carpus": (8, 9, 10),
    "right_elbow": (11, 12, 13), "right_carpus": (12, 13, 14),
    "left_stifle": (17, 18, 19), "left_hock": (18, 19, 20),
    "right_stifle": (21, 22, 23), "right_hock": (22, 23, 24),
}
LIMBS = {"left_fore": (7, 10), "right_fore": (11, 14), "left_hind": (17, 20), "right_hind": (21, 24)}
_HIP, _SHOULDER = 1, 6
_NECK, _HEAD = 15, 16


def posed_joints(rest_joints: np.ndarray, parents: np.ndarray, pose: np.ndarray, chunk: int = 256) -> np.ndarray:
    """(N, J, 3) joints in the body's own frame (no global rotation), from local rotations (N, J-1, 3, 3)."""
    rest = torch.as_tensor(rest_joints, dtype=torch.float32)
    par = torch.as_tensor(np.asarray(parents).astype(np.int64))
    out = []
    for start in range(0, len(pose), chunk):
        p = torch.as_tensor(pose[start:start + chunk], dtype=torch.float32)
        rotations = torch.cat([torch.eye(3).expand(len(p), 1, 3, 3), p], dim=1)
        out.append(batch_rigid_transform(rotations, rest.expand(len(p), -1, -1), par)[0].numpy())
    return np.concatenate(out)


def _angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)).clip(1e-9)
    return np.degrees(np.arccos(cos.clip(-1.0, 1.0)))


def joint_angles(joints: np.ndarray) -> pd.DataFrame:
    """Per frame: inner angle at each leg joint, and each limb's swing angle (degrees)."""
    columns = {name: _angle(joints[:, p] - joints[:, j], joints[:, c] - joints[:, j]) for name, (p, j, c) in JOINTS.items()}
    # Body axes per frame: forward along hip -> shoulder, up perpendicular to it in the side plane.
    forward = joints[:, _SHOULDER] - joints[:, _HIP]
    forward /= np.linalg.norm(forward, axis=-1, keepdims=True)
    side = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    side /= np.linalg.norm(side, axis=-1, keepdims=True).clip(1e-9)
    up = np.cross(side, forward)
    for name, (top, foot) in LIMBS.items():
        leg = joints[:, foot] - joints[:, top]
        columns[name] = np.degrees(np.arctan2((leg * forward).sum(-1), -(leg * up).sum(-1)))
    neck = joints[:, _HEAD] - joints[:, _NECK]
    columns["neck"] = np.degrees(np.arctan2((neck * up).sum(-1), (neck * forward).sum(-1)))
    return pd.DataFrame(columns)


def _runs(frames: np.ndarray, max_gap: int = 1) -> list[np.ndarray]:
    """Indices of consecutive stretches, allowing gaps of up to `max_gap` missing frames."""
    order = np.argsort(frames)
    breaks = np.where(np.diff(frames[order]) > max_gap + 1)[0] + 1
    return [r for r in np.split(order, breaks)]


def stride_frequency(signal: np.ndarray, fps: float, low_hz: float = 0.3, high_hz: float | None = None) -> tuple[float, float]:
    """Dominant swing frequency (Hz) and its share of the signal's power in that band, via the periodogram."""
    x = signal - signal.mean()
    n = len(x)
    if n < 8 or not np.isfinite(x).all():
        return float("nan"), 0.0
    window = np.hanning(n)
    power = np.abs(np.fft.rfft(x * window, n=4 * n)) ** 2
    freqs = np.fft.rfftfreq(4 * n, 1.0 / fps)
    high_hz = high_hz or fps / 2
    band = (freqs >= low_hz) & (freqs <= high_hz)
    if not band.any() or power[band].sum() <= 0:
        return float("nan"), 0.0
    peak = np.argmax(np.where(band, power, 0))
    near = band & (np.abs(freqs - freqs[peak]) <= 2 * fps / n)  # the peak's own width, for a 4x padded transform
    return float(freqs[peak]), float(power[near].sum() / power[band].sum())


def _periodogram(x: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    x = x - x.mean()
    n = len(x)
    return np.fft.rfftfreq(4 * n, 1.0 / fps), np.abs(np.fft.rfft(x * np.hanning(n), n=4 * n)) ** 2


def stride_frequency_lr(g: pd.DataFrame, fps: float, low_hz: float = 0.25, high_hz: float = 2.5) -> tuple[float, float]:
    """Stride frequency from left minus right swing of both limb pairs (powers summed), and the peak's share of power."""
    if len(g) < 8:
        return float("nan"), 0.0
    freqs, p_fore = _periodogram(g["left_fore"].to_numpy() - g["right_fore"].to_numpy(), fps)
    _, p_hind = _periodogram(g["left_hind"].to_numpy() - g["right_hind"].to_numpy(), fps)
    power = p_fore + p_hind
    band = (freqs >= low_hz) & (freqs <= min(high_hz, fps / 2))
    if not band.any() or power[band].sum() <= 0:
        return float("nan"), 0.0
    peak = np.argmax(np.where(band, power, 0))
    near = band & (np.abs(freqs - freqs[peak]) <= 2 * fps / len(g))
    return float(freqs[peak]), float(power[near].sum() / power[band].sum())


def _phase(x: np.ndarray, t: np.ndarray, freq: float) -> float:
    """Phase (radians) of the peak of x's component at `freq`: x ~ cos(2 pi f t - phase)."""
    return -float(np.angle(np.sum((x - x.mean()) * np.exp(-2j * np.pi * freq * t))))


def limb_phase(g: pd.DataFrame, t: np.ndarray, freq: float) -> float:
    """Fraction of a stride by which each forelimb's peak protraction follows the ipsilateral hindlimb's (mean of sides)."""
    lags = [((_phase(g[f"{side}_fore"].to_numpy(), t, freq) - _phase(g[f"{side}_hind"].to_numpy(), t, freq)) / (2 * np.pi)) % 1.0
            for side in ("left", "right")]
    return float(np.angle(np.mean(np.exp(2j * np.pi * np.array(lags)))) / (2 * np.pi) % 1.0)


def duty_factor(g: pd.DataFrame, t: np.ndarray, freq: float, harmonics: int = 3) -> float:
    """Fraction of time the limbs retract (rotate backwards relative to the body), on swing smoothed to `harmonics`."""
    fractions = []
    for name in LIMBS:
        x = g[name].to_numpy() - g[name].mean()
        smooth = np.zeros_like(t)
        for k in range(1, harmonics + 1):
            c = 2 * np.mean(x * np.exp(-2j * np.pi * k * freq * t))
            smooth += np.real(c * np.exp(2j * np.pi * k * freq * t))
        fractions.append(float(np.mean(np.diff(smooth) < 0)))
    return float(np.mean(fractions))


def ground_speed(g: pd.DataFrame, t: np.ndarray) -> float | None:
    """Speed (m/s) from a straight-line fit of the ground-plane position over the stretch, which averages out jitter."""
    if "x_m" not in g or g["x_m"].isna().any() or len(g) < 5:
        return None
    vx = np.polyfit(t, g["x_m"].to_numpy(), 1)[0]
    vz = np.polyfit(t, g["z_m"].to_numpy(), 1)[0]
    return float(np.hypot(vx, vz))


def walking_bout(g: pd.DataFrame, fps: float, min_speed: float = 0.3, window: int = 21, min_frames: int = 30) -> pd.DataFrame | None:
    """Longest stretch where the animal moves at `min_speed`+ (straight-line fit over `window` frames), if long enough."""
    if "x_m" not in g or g["x_m"].isna().any() or len(g) < max(window, min_frames):
        return None
    t = (g["frame"].to_numpy() - g["frame"].iloc[0]) / fps
    half = window // 2
    moving = np.zeros(len(g), dtype=bool)
    for i in range(half, len(g) - half):
        sl = slice(i - half, i + half + 1)
        moving[i] = ground_speed(g.iloc[sl], t[sl]) >= min_speed
    edges = np.flatnonzero(np.diff(np.r_[0, moving.astype(int), 0]))
    if not len(edges):
        return None
    starts, ends = edges[::2], edges[1::2]
    k = int(np.argmax(ends - starts))
    a, b = max(starts[k] - half, 0), min(ends[k] + half, len(g))
    return g.iloc[a:b] if b - a >= min_frames else None


def gait_summary(table: pd.DataFrame, fps: float, min_frames: int = 20) -> list[dict]:
    """Per track, over its longest consecutive stretch of usable frames, narrowed to its longest walking bout when it has
    one (see walking_bout): speed, stride frequency, limb phase, duty factor and joint ranges.

    `table` needs columns frame, track_id, usable, speed_m_s (may be NaN), and the angle columns."""
    rows = []
    for track, group in table[table.usable].groupby("track_id"):
        frames = group["frame"].to_numpy()
        run = max(_runs(frames), key=len)
        g = group.iloc[run].sort_values("frame")
        if len(g) < min_frames:
            continue
        bout = walking_bout(g, fps)
        stretch_frames = len(g)
        if bout is not None:
            g = bout
        t = (g["frame"].to_numpy() - g["frame"].iloc[0]) / fps
        freq, share = stride_frequency_lr(g, fps)
        old_freq, old_share = stride_frequency(g["left_fore"].to_numpy() - g["left_hind"].to_numpy(), fps)
        speed = ground_speed(g, t)
        row = {"track_id": int(track), "frames": len(g), "first_frame": int(g.frame.iloc[0]), "last_frame": int(g.frame.iloc[-1]),
               "walking_bout": bout is not None, "stretch_frames": stretch_frames,
               "speed_m_s": speed,
               "speed_frame_median_m_s": float(np.nanmedian(g["speed_m_s"])) if g["speed_m_s"].notna().any() else None,
               "stride_hz": freq, "stride_periodicity": share,
               "stride_hz_fore_minus_hind": old_freq, "stride_periodicity_fore_minus_hind": old_share,
               "stride_length_m": speed / freq if speed is not None and np.isfinite(freq) and freq > 0 else None,
               "limb_phase": limb_phase(g, t, freq) if np.isfinite(freq) else None,
               "duty_factor": duty_factor(g, t, freq) if np.isfinite(freq) else None,
               "shoulder_height_m": float(g["shoulder_height_m"].iloc[0]) if "shoulder_height_m" in g and g["shoulder_height_m"].notna().any() else None}
        # Left and right hind swing in antiphase for walk and trot: the correlation says how gait-like the motion is.
        row["left_right_hind_corr"] = float(np.corrcoef(g["left_hind"], g["right_hind"])[0, 1])
        for name in list(JOINTS) + list(LIMBS) + (["neck"] if "neck" in g else []):
            values = g[name].to_numpy()
            row[f"{name}_median"] = float(np.median(values))
            row[f"{name}_range"] = float(np.percentile(values, 95) - np.percentile(values, 5))
        rows.append(row)
    return rows
