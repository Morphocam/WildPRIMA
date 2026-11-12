"""A SMAL body whose proportions can leave SMAL's linear shape space, for any quadruped.

SMAL's 41 shape components were learned from toy scans of cats, dogs, horses, cows and hippos, so plausible
coefficients cannot produce every species (a giraffe, say). Two additions make other bodies reachable:

* bone scales: each group of bones has a learnable length and thickness, applied to the skeleton and
  to the vertices each bone skins, which is how a neck can become several times longer;
* vertex offsets: left-right symmetric per-vertex displacements for detail (as in SMAL+D);
* humps: one or two domes on the back (camels), appended like the horns and skinned like the back beneath;
* horns: optional tapered tubes appended after SMAL's vertices and skinned to the head, either a mirrored pair
  (cattle, antelope) or one or two on the midline of the snout (rhinos).

Nothing here is species-specific. The limits only rule out impossible bodies; `animal3d.fit` measures where
within them each animal sits, and whether it has horns at all. SMAL's own vertices, joints and skinning are
kept, so the body stays posable by PRIMA's pose. Horns need no joints of their own because they are rigid on the
skull. Without them, the ears and the head are what stretch to cover a horn, so the ears are held tightly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from smplx.lbs import batch_rigid_transform
from torch import nn

# Joint indices follow `name2id35` in prima/models/smal_wrapper.py.
BONE_GROUPS = {
    "torso": (0, 1, 2, 3, 4, 5, 6),
    "neck": (15,),
    "head": (16, 32),
    "ears": (33, 34),
    "fore_upper": (7, 11),
    "fore_mid": (8, 12),
    "fore_lower": (9, 13),
    "fore_foot": (10, 14),
    "hind_upper": (17, 21),
    "hind_mid": (18, 22),
    "hind_lower": (19, 23),
    "hind_foot": (20, 24),
    "tail": (25, 26, 27, 28, 29, 30, 31),
}
# Limits as (length min, length max, thickness min, thickness max), wide enough for giraffe to rhino.
# Torso length is fixed at 1: from images alone, overall size trades off freely against camera depth, so
# one length must be held for the others to mean anything. True size comes from LiDAR, afterwards.
BODY_BOUNDS = {
    "torso": (1.0, 1.0, 0.6, 1.5),
    "neck": (0.6, 5.0, 0.35, 1.5),
    "head": (0.6, 1.6, 0.5, 1.4),
    "ears": (0.6, 1.2, 0.6, 1.2),
    "fore_upper": (0.6, 2.2, 0.4, 1.5),
    "fore_mid": (0.6, 2.8, 0.35, 1.5),
    "fore_lower": (0.6, 2.8, 0.35, 1.5),
    "fore_foot": (0.6, 2.0, 0.4, 1.5),
    "hind_upper": (0.6, 2.2, 0.4, 1.5),
    "hind_mid": (0.6, 2.8, 0.35, 1.5),
    "hind_lower": (0.6, 2.8, 0.35, 1.5),
    "hind_foot": (0.6, 2.0, 0.4, 1.5),
    "tail": (0.5, 2.0, 0.3, 1.5),
}
LEG_GROUPS = tuple(name for name in BONE_GROUPS if name.startswith(("fore_", "hind_")))
# Bones with several children need a stated direction: root -> spine, head -> mouth.
# The spine top (6) branches to neck and both forelegs, so it continues its parent's direction.
_PRIMARY_CHILD = {0: 1, 6: None, 16: 32}
_PELVIS, _NECK, _HEAD, _MOUTH = 1, 15, 16, 32
_EARS = (33, 34)
_FORELEG, _HINDLEG = (7, 8, 9, 10), (17, 18, 19, 20)


def make_init(neck: float = 1.0, leg: float = 1.0, thin: float = 1.0) -> dict[str, tuple[float, float]]:
    """A starting body: (length, thickness) per bone group, from three overall proportions."""
    init = {name: (leg, thin) for name in LEG_GROUPS}
    init["neck"] = (neck, thin)
    return init


def _bone_directions(joints: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
    """Unit direction along each joint's bone; joints with no clear child inherit their parent's."""
    n = len(parents)
    children: dict[int, list[int]] = {j: [] for j in range(n)}
    for j in range(1, n):
        children[int(parents[j])].append(j)
    directions = torch.zeros_like(joints)
    for j in range(n):  # Parents have lower indices than children, so theirs are already known.
        child = _PRIMARY_CHILD.get(j, children[j][0] if len(children[j]) == 1 else None)
        if child is not None:
            offset = joints[child] - joints[j]
            directions[j] = offset / offset.norm().clamp(min=1e-8)
        elif j > 0:
            directions[j] = directions[int(parents[j])]
        else:
            directions[j] = torch.tensor([1.0, 0.0, 0.0], dtype=joints.dtype, device=joints.device)
    return directions


def mirror_indices(vertices: np.ndarray) -> np.ndarray:
    """For each vertex, the nearest vertex to its reflection across the body's left-right plane."""
    from scipy.spatial import cKDTree

    return cKDTree(vertices).query(vertices * np.array([1.0, -1.0, 1.0]))[1]


@dataclass(frozen=True)
class HornShape:
    """Horns of one of two layouts. Angles in degrees, sizes in model units.

    "paired" (cattle, goats, antelope): one horn and its mirror image. It starts at `elevation` above horizontal, and
    `sweep` forward of pointing straight out sideways. Along its length it turns by `curl_up` and `curl_forward`. The
    base sits at the ear joint, moved by `base_inward`, `base_forward` and `base_up`.

    "nasal" (rhinos): horns on the midline of the nose. The front one stands on the top of the snout at
    `nose_position` (in head lengths from the head joint towards the mouth; the snout tip is near 2), leaning
    `elevation` up from the snout's line and curving back by `curl_up`. A second horn, `rear_length` times as long,
    stands `rear_gap` head lengths behind it; `rear_length` 0 means a single horn.
    """

    length: float = 0.4
    radius: float = 0.05
    elevation: float = 20.0
    sweep: float = 0.0
    curl_up: float = 0.0
    curl_forward: float = 0.0
    base_inward: float = 0.03
    base_forward: float = 0.0
    base_up: float = 0.02
    nose_position: float = 1.75
    rear_length: float = 0.0
    rear_gap: float = 0.45
    layout: str = "paired"


# (min, max) for each field. Length has to reach from a short horn to an Ankole's. Radius is kept slender, because a
# fat tube can fill any gap in an outline, which makes it fit better while looking nothing like a horn.
HORN_BOUNDS = {
    "length": (0.05, 1.5), "radius": (0.01, 0.07), "elevation": (-90.0, 120.0), "sweep": (-90.0, 90.0),
    "curl_up": (-180.0, 180.0), "curl_forward": (-180.0, 180.0),
    "base_inward": (-0.05, 0.12), "base_forward": (-0.12, 0.12), "base_up": (-0.08, 0.12),
    "nose_position": (1.75, 1.75), "rear_length": (0.0, 0.0), "rear_gap": (0.45, 0.45),
}
# Nose horns are cones, broad at the base, so they may be thicker. Sideways fields are fixed at zero.
NASAL_BOUNDS = {
    **{name: (0.0, 0.0) for name in ("sweep", "curl_forward", "base_inward", "base_forward")},
    "length": (0.03, 0.8), "radius": (0.015, 0.1), "elevation": (30.0, 130.0), "curl_up": (-30.0, 90.0),
    "base_up": (-0.04, 0.04), "nose_position": (1.3, 2.0), "rear_length": (0.0, 0.8), "rear_gap": (0.25, 0.7),
}
SINGLE_NASAL_BOUNDS = {**NASAL_BOUNDS, "rear_length": (0.0, 0.0)}
_HORN_FIELDS = tuple(HORN_BOUNDS)
_HORN_RINGS, _HORN_SIDES = 10, 8
# An optimizer step should move a horn about as much whichever field it changes, so angles work in units of 50 degrees.
_HORN_UNITS = {"elevation": 50.0, "sweep": 50.0, "curl_up": 50.0, "curl_forward": 50.0}


def snout_profile(vertices: np.ndarray, joints: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vertices along the top of the head's midline, from the head joint to the snout tip, and their positions in
    head lengths. Nose horns stand on this line; taking it from the surface keeps them on the skin as the head changes.
    """
    forward = joints[_MOUTH] - joints[_HEAD]
    forward[1] = 0.0
    length = np.linalg.norm(forward)
    forward /= length
    up = np.cross(forward, [0.0, 1.0, 0.0])
    up *= np.sign(up[2])
    t = (vertices - joints[_HEAD]) @ forward / length
    height = (vertices - joints[_HEAD]) @ up
    candidate = ((weights[:, _HEAD] + weights[:, _MOUTH]) > 0.5) & (np.abs(vertices[:, 1]) < 0.015) & (t >= 0.0)
    picks, positions = [], []
    for start in np.arange(0.0, t[candidate].max(), 0.1):
        band = np.where(candidate & (t >= start) & (t < start + 0.1))[0]
        if len(band):
            top = band[np.argmax(height[band])]
            picks.append(top)
            positions.append(t[top])
    picks, positions = np.array(picks), np.array(positions)
    # Drop dips (nostrils, the mouth's corner): keep only points near the line through their neighbours' tops.
    keep = np.ones(len(picks), dtype=bool)
    for i in range(1, len(picks) - 1):
        expected = np.interp(positions[i], positions[[i - 1, i + 1]], height[picks[[i - 1, i + 1]]])
        keep[i] = height[picks[i]] > expected - 0.03
    return picks[keep], positions[keep]


class Horns(nn.Module):
    """Tapered, curved tubes built from the head's rest position: a mirrored pair, or up to two on the nose."""

    n_vertices = 2 * _HORN_RINGS * _HORN_SIDES

    def __init__(self, shape: HornShape, profile: tuple[np.ndarray, np.ndarray] | None = None):
        super().__init__()
        self.layout = "nasal" if shape.layout.startswith("nasal") else "paired"
        bounds = {"paired": HORN_BOUNDS, "nasal": NASAL_BOUNDS, "nasal_one": SINGLE_NASAL_BOUNDS}.get(shape.layout, NASAL_BOUNDS)
        unit = lambda name: _HORN_UNITS.get(name, 1.0)
        values = [getattr(shape, name) / unit(name) for name in _HORN_FIELDS]
        self.raw = nn.Parameter(torch.tensor(values, dtype=torch.float32))
        self.register_buffer("low", torch.tensor([bounds[n][0] / unit(n) for n in _HORN_FIELDS], dtype=torch.float32))
        self.register_buffer("high", torch.tensor([bounds[n][1] / unit(n) for n in _HORN_FIELDS], dtype=torch.float32))
        self.register_buffer("units", torch.tensor([unit(n) for n in _HORN_FIELDS], dtype=torch.float32))
        self.shape_layout = shape.layout
        if self.layout == "nasal":
            if profile is None:
                raise ValueError("nose horns need the snout profile")
            self.register_buffer("profile_index", torch.as_tensor(profile[0], dtype=torch.long))
            self.register_buffer("profile_t", torch.as_tensor(profile[1], dtype=torch.float32))

    def values(self) -> dict[str, torch.Tensor]:
        clamped = torch.max(torch.min(self.raw, self.high), self.low) * self.units
        return dict(zip(_HORN_FIELDS, clamped))

    def shape(self) -> HornShape:
        return HornShape(**{name: float(value) for name, value in self.values().items()}, layout=self.shape_layout)

    def mirror(self) -> torch.Tensor:
        """For each horn vertex, the horn vertex at its left-right reflection (indices from 0)."""
        n, half = self.n_vertices, self.n_vertices // 2
        if self.layout == "paired":
            return torch.cat([torch.arange(half, n), torch.arange(half)])
        k = torch.arange(n)
        ring_start, around = k - k % _HORN_SIDES, k % _HORN_SIDES
        return ring_start + (_HORN_SIDES - around) % _HORN_SIDES  # a midline tube is its own mirror image

    @staticmethod
    def _tube(base, up, length, radius, elevation, curl_up, sweep, curl_forward, side, forward):
        device, dtype = base.device, base.dtype
        s = torch.linspace(0.0, 1.0, _HORN_RINGS, device=device, dtype=dtype)
        elevation = torch.deg2rad(elevation + curl_up * s)
        sweep = torch.deg2rad(sweep + curl_forward * s)
        out = torch.cos(sweep)[:, None] * side + torch.sin(sweep)[:, None] * forward
        tangent = torch.cos(elevation)[:, None] * out + torch.sin(elevation)[:, None] * up
        normal_a = -torch.sin(elevation)[:, None] * out + torch.cos(elevation)[:, None] * up
        normal_b = -torch.sin(sweep)[:, None] * side + torch.cos(sweep)[:, None] * forward
        steps = tangent * length / (_HORN_RINGS - 1)
        centre = base + torch.cat([torch.zeros_like(steps[:1]), torch.cumsum(steps[:-1], dim=0)])
        r = radius * (1.0 - 0.92 * s)
        angle = torch.linspace(0.0, 2 * np.pi, _HORN_SIDES + 1, device=device, dtype=dtype)[:-1]
        around = (torch.cos(angle)[None, :, None] * normal_a[:, None] + torch.sin(angle)[None, :, None] * normal_b[:, None])
        return (centre[:, None] + r[:, None, None] * around).reshape(-1, 3)

    def _on_snout(self, surface: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        """The point on the snout's top line at `position` head lengths, by linear interpolation."""
        t = self.profile_t
        i = torch.clamp(torch.searchsorted(t, position.detach().reshape(1)), 1, len(t) - 1)[0]
        w = ((position - t[i - 1]) / (t[i] - t[i - 1])).clamp(0.0, 1.0)
        points = surface[self.profile_index]
        point = (1 - w) * points[i - 1] + w * points[i]
        return point * torch.tensor([1.0, 0.0, 1.0], device=point.device, dtype=point.dtype)  # exactly on the midline

    def vertices(self, joints: torch.Tensor, surface: torch.Tensor | None = None) -> torch.Tensor:
        """(n_vertices, 3). Paired: one horn's rings, then its mirror image (vertex i + n/2 mirrors vertex i).
        Nasal: the front horn, then the rear one. `surface` is the body's rest vertices, needed for nose horns."""
        v = self.values()
        device, dtype = joints.device, joints.dtype
        side = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
        snout = joints[_MOUTH] - joints[_HEAD]
        forward = snout - (snout @ side) * side
        forward = forward / forward.norm().clamp(min=1e-8)
        up = torch.linalg.cross(forward, side)
        if up[2] < 0:
            up = -up

        if self.layout == "paired":
            ear = joints[_EARS[0]] if joints[_EARS[0], 1] > 0 else joints[_EARS[1]]
            base = ear - v["base_inward"] * side + v["base_forward"] * forward + v["base_up"] * up
            rings = self._tube(base, up, v["length"], v["radius"], v["elevation"], v["curl_up"], v["sweep"],
                               v["curl_forward"], side, forward)
            return torch.cat([rings, rings * torch.tensor([1.0, -1.0, 1.0], device=device, dtype=dtype)])

        # Sweep 90 points a tube forward along the midline; elevation then tilts it up from the snout's line.
        quarter = torch.tensor(90.0, device=device, dtype=dtype)
        zero = torch.zeros((), device=device, dtype=dtype)
        front_base = self._on_snout(surface, v["nose_position"]) + v["base_up"] * up
        front = self._tube(front_base, up, v["length"], v["radius"], v["elevation"], v["curl_up"], quarter, zero,
                           side, forward)
        rear_base = self._on_snout(surface, v["nose_position"] - v["rear_gap"]) + v["base_up"] * up
        rear_length = v["rear_length"] * v["length"]
        # A missing rear horn collapses to a point inside the skin, so the mesh keeps one size for every layout.
        rear_radius = v["radius"] * torch.clamp(v["rear_length"] / 0.4, max=1.0) * 0.9
        rear = self._tube(rear_base - 0.01 * up, up, rear_length, rear_radius, v["elevation"] + 10.0, v["curl_up"],
                          quarter, zero, side, forward)
        return torch.cat([front, rear])

    def faces(self, first_vertex: int) -> np.ndarray:
        """Triangles of both tubes, open at the base (it sits inside the head) and closed at the tip."""
        half = _HORN_RINGS * _HORN_SIDES
        quads = []
        for ring in range(_HORN_RINGS - 1):
            for k in range(_HORN_SIDES):
                a, b = ring * _HORN_SIDES + k, ring * _HORN_SIDES + (k + 1) % _HORN_SIDES
                c, d = a + _HORN_SIDES, b + _HORN_SIDES
                quads += [(a, b, d), (a, d, c)]
        tip = (_HORN_RINGS - 1) * _HORN_SIDES
        quads += [(tip, tip + k, tip + k + 1) for k in range(1, _HORN_SIDES - 1)]
        one = np.array(quads, dtype=np.int64)
        # A mirrored copy turns inside out unless its triangles are reversed; a second nose horn is not mirrored.
        second = one[:, ::-1] if self.layout == "paired" else one
        return np.concatenate([one + first_vertex, second + half + first_vertex])


_HIP, _SHOULDER = 1, 6
_TORSO_JOINTS = (0, 1, 2, 3, 4, 5, 6)
# Per hump: position (0 at the hip, 1 at the shoulder), height above the back (model units), half-length along the back
# (fraction of the hip-shoulder span) and half-width across it (model units). A Bactrian camel's humps rise about 0.3 m
# on a body about 1.5 m long, which is about 0.2 units here; the torso is about 0.34 units wide.
# The size limits matter: left free, both domes grew to fill the whole back (they pad a furry outline), overlapping into
# one wide mass. Humps are narrower than the torso, and two humps' footprints may at most touch.
HUMP_BOUNDS = {"height": (0.0, 0.3), "length": (0.08, 0.22), "width": (0.04, 0.11)}
# Two humps keep to their own ends of the back (over the loin and the withers), with a dip between.
HUMP_POSITIONS = {1: [(0.2, 0.8)], 2: [(-0.05, 0.3), (0.7, 1.05)]}
HUMP_STARTS = {1: [0.5], 2: [0.2, 0.8]}
_HUMP_RINGS, _HUMP_SIDES = 10, 16
_HUMP_SINK = 0.06  # how far each dome's base reaches below the skin, so no gap opens at its edge


def back_profile(vertices: np.ndarray, joints: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vertices along the top of the back's midline and their positions (0 at the hip, 1 at the shoulder). Humps
    stand on this line; reading it from the surface keeps them on the skin as the torso changes thickness."""
    hip, shoulder = joints[_HIP], joints[_SHOULDER]
    along = (vertices[:, 0] - hip[0]) / (shoulder[0] - hip[0])
    spine_z = hip[2] + np.clip(along, 0, 1) * (shoulder[2] - hip[2])
    torso = weights[:, list(_TORSO_JOINTS)].sum(axis=1) > 0.5
    candidate = torso & (np.abs(vertices[:, 1]) < 0.015) & (vertices[:, 2] > spine_z)
    picks, positions = [], []
    for start in np.arange(-0.2, 1.2, 0.1):
        band = np.where(candidate & (along >= start) & (along < start + 0.1))[0]
        if len(band):
            top = band[np.argmax(vertices[band, 2])]
            picks.append(top)
            positions.append(along[top])
    order = np.argsort(positions)
    return np.array(picks)[order], np.array(positions)[order]


class Humps(nn.Module):
    """One or two domes on the back (camels): their own geometry, like the horns, so two humps stay two humps
    however few vertices the template has there. Each is a half-ellipsoid standing on the back's midline with its base
    sunk into the body, skinned like the back beneath it. Height 0 leaves only a flat cap under the skin."""

    per_hump = 1 + _HUMP_RINGS * _HUMP_SIDES

    def __init__(self, count: int, profile: tuple[np.ndarray, np.ndarray], height: float = 0.12, length: float = 0.18,
                 width: float = 0.09):
        super().__init__()
        if count not in HUMP_POSITIONS:
            raise ValueError("humps: 1 or 2")
        self.count = count
        self.n_vertices = count * self.per_hump
        start = [[p, height, length, width] for p in HUMP_STARTS[count]]
        self.raw = nn.Parameter(torch.tensor(start, dtype=torch.float32))
        low = [[lo, HUMP_BOUNDS["height"][0], HUMP_BOUNDS["length"][0], HUMP_BOUNDS["width"][0]] for lo, _ in HUMP_POSITIONS[count]]
        high = [[hi, HUMP_BOUNDS["height"][1], HUMP_BOUNDS["length"][1], HUMP_BOUNDS["width"][1]] for _, hi in HUMP_POSITIONS[count]]
        self.register_buffer("low", torch.tensor(low))
        self.register_buffer("high", torch.tensor(high))
        self.register_buffer("profile_index", torch.as_tensor(profile[0], dtype=torch.long))
        self.register_buffer("profile_t", torch.as_tensor(profile[1], dtype=torch.float32))

    def values(self) -> torch.Tensor:
        """(count, 4): position, height, half-length, half-width, within their limits."""
        return torch.max(torch.min(self.raw, self.high), self.low)

    def describe(self) -> list[dict]:
        return [dict(zip(("position", "height", "length", "width"), map(float, row))) for row in self.values()]

    def _blend(self, position: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Neighbouring back-profile entries and their interpolation weight at `position`."""
        t = self.profile_t
        i = torch.clamp(torch.searchsorted(t, position.detach().reshape(1)), 1, len(t) - 1)[0]
        w = ((position - t[i - 1]) / (t[i] - t[i - 1])).clamp(0.0, 1.0)
        return i, w

    def vertices(self, surface: torch.Tensor, joints: torch.Tensor) -> torch.Tensor:
        """(n_vertices, 3): per hump, its top vertex, then rings from the top down, each going round the hump."""
        device, dtype = surface.device, surface.dtype
        axis = joints[_SHOULDER] - joints[_HIP]
        axis = axis * torch.tensor([1.0, 0.0, 1.0], device=device, dtype=dtype)
        span = axis.norm()
        forward = axis / span.clamp(min=1e-8)
        side = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
        up = torch.linalg.cross(side, forward)
        if up[2] < 0:
            up = -up
        # Polar angle from the top: the last ring is a little below the ellipsoid's equator, inside the body.
        theta = torch.linspace(0.0, 1.0, _HUMP_RINGS + 1, device=device, dtype=dtype)[1:] * np.deg2rad(100.0)
        phi = torch.linspace(0.0, 2 * np.pi, _HUMP_SIDES + 1, device=device, dtype=dtype)[:-1]
        sin_t, cos_t = torch.sin(theta)[:, None, None], torch.cos(theta)[:, None, None]
        parts = []
        points = surface[self.profile_index]
        for position, height, length, width in self.values():
            i, w = self._blend(position)
            base = ((1 - w) * points[i - 1] + w * points[i]) * torch.tensor([1.0, 0.0, 1.0], device=device, dtype=dtype)
            vertical = height + _HUMP_SINK
            centre = base - _HUMP_SINK * up
            ring = (length * span * torch.cos(phi)[None, :, None] * forward + width * torch.sin(phi)[None, :, None] * side)
            rings = centre + sin_t.pow(1.4) * ring + vertical * cos_t * up  # narrower near the top: a peaked dome
            parts += [(centre + vertical * up)[None], rings.reshape(-1, 3)]
        return torch.cat(parts)

    @torch.no_grad()
    def skin(self, body_weights: torch.Tensor) -> torch.Tensor:
        """(n_vertices, J): each hump moves like the back beneath its centre."""
        rows = []
        table = body_weights[self.profile_index]
        for position in self.values()[:, 0]:
            i, w = self._blend(position)
            rows.append(((1 - w) * table[i - 1] + w * table[i]).expand(self.per_hump, -1))
        return torch.cat(rows)

    def mirror(self) -> torch.Tensor:
        """For each hump vertex, the vertex at its left-right reflection (indices from 0): each is its own mirror."""
        k = torch.arange(self.per_hump - 1)
        ring_start, around = k - k % _HUMP_SIDES, k % _HUMP_SIDES
        one = torch.cat([torch.zeros(1, dtype=torch.long), 1 + ring_start + (_HUMP_SIDES - around) % _HUMP_SIDES])
        return torch.cat([one + h * self.per_hump for h in range(self.count)])

    def faces(self, first_vertex: int) -> np.ndarray:
        """Triangles of every dome, facing outwards, open at the base (it sits inside the body)."""
        tris = [(0, 1 + (k + 1) % _HUMP_SIDES, 1 + k) for k in range(_HUMP_SIDES)]
        for ring in range(_HUMP_RINGS - 1):
            for k in range(_HUMP_SIDES):
                a, b = 1 + ring * _HUMP_SIDES + k, 1 + ring * _HUMP_SIDES + (k + 1) % _HUMP_SIDES
                c, d = a + _HUMP_SIDES, b + _HUMP_SIDES
                tris += [(a, b, d), (a, d, c)]
        one = np.array(tris, dtype=np.int64)[:, ::-1]  # built clockwise seen from outside, so reversed
        return np.concatenate([one + first_vertex + h * self.per_hump for h in range(self.count)])


class AnimalBody(nn.Module):
    def __init__(self, v_template, shapedirs, j_regressor, parents, weights, faces, betas, init=None, groups=None, bounds=None,
                 horns: HornShape | None = None, humps: int = 0):
        super().__init__()
        as_float = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)
        self.register_buffer("v_template", as_float(v_template))
        self.register_buffer("shapedirs", as_float(shapedirs))
        self.register_buffer("j_regressor", as_float(j_regressor))
        self.register_buffer("parents", torch.as_tensor(np.asarray(parents).astype(np.int64)))
        self.faces = np.asarray(faces)
        self.register_buffer("mirror", torch.as_tensor(mirror_indices(np.asarray(v_template))))

        groups = BONE_GROUPS if groups is None else groups
        # Offsets would otherwise pull the ears out over a horn, wherever the bone limits stop them.
        skin = as_float(weights)
        self.register_buffer("offset_freedom", (1.0 - skin[:, list(groups.get("ears", ()))].sum(dim=1))[:, None])
        self.group_names = list(groups)
        joint_group = torch.zeros(len(parents), dtype=torch.long)
        for g, name in enumerate(self.group_names):
            joint_group[list(groups[name])] = g
        self.register_buffer("joint_group", joint_group)

        init = {} if init is None else init
        scales = np.array([init.get(name, (1.0, 1.0)) for name in self.group_names], dtype=np.float32)
        self.log_scale = nn.Parameter(torch.log(torch.as_tensor(scales)))
        bounds = {} if bounds is None else bounds
        limits = np.array([bounds.get(name, (1e-3, 1e3, 1e-3, 1e3)) for name in self.group_names], dtype=np.float32)
        self.register_buffer("log_scale_min", torch.log(torch.as_tensor(limits[:, [0, 2]])))
        self.register_buffer("log_scale_max", torch.log(torch.as_tensor(limits[:, [1, 3]])))
        self.betas = nn.Parameter(torch.as_tensor(np.asarray(betas), dtype=torch.float32))
        self.offsets = nn.Parameter(torch.zeros_like(self.v_template))

        n_verts = self.v_template.shape[0]
        edges = np.concatenate([self.faces[:, [0, 1]], self.faces[:, [1, 2]], self.faces[:, [2, 0]]]).astype(np.int64)
        self.register_buffer("edges", torch.as_tensor(edges))
        self.n_verts = n_verts

        self.horns = None
        if horns is not None:
            profile = None
            if horns.layout != "paired":
                template = np.asarray(v_template)
                profile = snout_profile(template, np.asarray(j_regressor) @ template, np.asarray(weights))
            self.horns = Horns(horns, profile)
            n = Horns.n_vertices
            head_skin = torch.zeros(n, skin.shape[1])
            head_skin[:, _HEAD] = 1.0
            skin = torch.cat([skin, head_skin])
            self.faces = np.concatenate([self.faces, self.horns.faces(n_verts)])
            self.mirror = torch.cat([self.mirror, self.horns.mirror() + n_verts])
        self.humps = None
        if humps:
            template = np.asarray(v_template)
            self.humps = Humps(humps, back_profile(template, np.asarray(j_regressor) @ template, np.asarray(weights)))
            first = len(skin)
            skin = torch.cat([skin, self.humps.skin(skin[:n_verts])])
            self.faces = np.concatenate([self.faces, self.humps.faces(first)])
            self.mirror = torch.cat([self.mirror, self.humps.mirror() + first])
        self.register_buffer("weights", skin)

    def symmetric_offsets(self) -> torch.Tensor:
        mirrored = self.offsets[self.mirror[: self.n_verts]] * torch.tensor([1.0, -1.0, 1.0], device=self.offsets.device)
        return 0.5 * (self.offsets + mirrored) * self.offset_freedom

    def scales(self) -> torch.Tensor:
        """(length, thickness) per bone group, within the limits."""
        return torch.exp(torch.max(torch.min(self.log_scale, self.log_scale_max), self.log_scale_min))

    def rest(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Neutral-pose vertices and joints, after shape, bone scales and offsets."""
        shaped = self.v_template + self.shapedirs @ self.betas
        joints = self.j_regressor @ shaped
        directions = _bone_directions(joints, self.parents)

        scale = self.scales()[self.joint_group]
        length, thick = scale[:, 0, None, None], scale[:, 1, None, None]
        # Scale along the bone by `length` and across it by `thick`.
        stretch = thick * torch.eye(3, device=joints.device) + (length - thick) * (
            directions[:, :, None] * directions[:, None, :]
        )

        new_joints = [joints[0]]
        for j in range(1, len(self.parents)):
            parent = int(self.parents[j])
            new_joints.append(new_joints[parent] + stretch[parent] @ (joints[j] - joints[parent]))
        new_joints = torch.stack(new_joints)

        # Each vertex follows its bones, blended by skinning weight: (V, J, 3).
        relative = shaped[:, None, :] - joints[None]
        moved = new_joints[None] + (stretch[None] @ relative[..., None])[..., 0]
        body = (self.weights[: self.n_verts, :, None] * moved).sum(dim=1) + self.symmetric_offsets()
        vertices = body
        if self.horns is not None:
            vertices = torch.cat([vertices, self.horns.vertices(new_joints, body)])
        if self.humps is not None:
            vertices = torch.cat([vertices, self.humps.vertices(body, new_joints)])
        return vertices, new_joints

    def skin(self) -> torch.Tensor:
        """Skinning weights for every vertex. A hump moves with the back under it, wherever the fit has put it."""
        if self.humps is None:
            return self.weights
        first = len(self.weights) - self.humps.n_vertices
        return torch.cat([self.weights[:first], self.humps.skin(self.weights[: self.n_verts])])

    def forward(self, global_orient: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        """Posed vertices (N, V, 3) from rotation matrices (N, 3, 3) and (N, J-1, 3, 3)."""
        rest_v, rest_j = self.rest()
        n = global_orient.shape[0]
        rotations = torch.cat([global_orient[:, None], pose], dim=1)
        _, transforms = batch_rigid_transform(rotations, rest_j.expand(n, -1, -1), self.parents)
        blended = (self.skin()[None, :, :, None, None] * transforms[:, None]).sum(dim=2)
        homogeneous = torch.cat([rest_v, torch.ones_like(rest_v[:, :1])], dim=1)
        return (blended @ homogeneous[None, :, :, None])[:, :, :3, 0]

    @torch.no_grad()
    def posed_joints(self, global_orient: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        """Joint positions (N, J, 3) in the same pose as `forward`."""
        _, rest_j = self.rest()
        rotations = torch.cat([global_orient[:, None], pose], dim=1)
        return batch_rigid_transform(rotations, rest_j.expand(len(pose), -1, -1), self.parents)[0]

    def laplacian_loss(self) -> torch.Tensor:
        """Keeps the offsets smooth, so they cannot crumple the surface to fit noisy mask edges."""
        offsets = self.symmetric_offsets()
        i, j = self.edges[:, 0], self.edges[:, 1]
        total = torch.zeros_like(offsets)
        count = torch.zeros(self.n_verts, device=offsets.device)
        total = total.index_add(0, i, offsets[j]).index_add(0, j, offsets[i])
        ones = torch.ones_like(i, dtype=offsets.dtype)
        count = count.index_add(0, i, ones).index_add(0, j, ones)
        return (offsets - total / count.clamp(min=1)[:, None]).pow(2).sum(dim=1).mean()

    @torch.no_grad()
    def measurements(self) -> dict[str, float]:
        """Bone-path lengths in model units, independent of pose. Multiply by metres per unit for metres."""
        _, j = self.rest()
        path = lambda chain: float(sum((j[a] - j[b]).norm() for a, b in zip(chain, chain[1:])))
        horn = {}
        if self.humps is not None:
            horn.update({f"hump{i + 1}_height": h["height"] for i, h in enumerate(self.humps.describe())})
        if self.horns is not None:
            values = self.horns.values()
            horn["horn"] = float(values["length"])
            if self.horns.layout == "nasal":
                horn["rear_horn"] = float(values["length"] * values["rear_length"])
        return {
            **horn,
            "body": float((j[_PELVIS] - j[_NECK]).norm()),
            "neck": float((j[_NECK] - j[_HEAD]).norm()),
            "head": float((j[_HEAD] - j[_MOUTH]).norm()),
            "foreleg": path(_FORELEG),
            "hindleg": path(_HINDLEG),
        }

    def proportions(self) -> dict[str, float]:
        """Ratios that describe the body plan, comparable with `PrimaModel.shape_stats`."""
        m = self.measurements()
        return {"neck_to_body": m["neck"] / m["body"], "leg_to_body": m["foreleg"] / m["body"]}
