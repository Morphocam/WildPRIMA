"""A SMAL body whose proportions can leave SMAL's linear shape space.

SMAL's 41 shape components were learned from toy scans of cats, dogs, horses, cows and hippos, so no
plausible coefficients produce a giraffe. Two additions on top of the linear model make it reachable:

* bone scales: each group of bones has a learnable length and thickness, applied to the skeleton and
  to the vertices each bone skins, which is how the neck can become several times longer;
* vertex offsets: left-right symmetric per-vertex displacements for detail (as in SMAL+D).

The result still has SMAL's topology, joints and skinning weights, so it stays posable.
"""
from __future__ import annotations

import numpy as np
import torch
from smplx.lbs import batch_rigid_transform
from torch import nn

# Joint indices follow `name2id35` in prima/models/smal_wrapper.py.
BONE_GROUPS = {
    "torso": (0, 1, 2, 3, 4, 5, 6),
    "neck": (15,),
    "head": (16, 32, 33, 34),
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
# (length, thickness) starting points from general giraffe anatomy; fitting refines them.
GIRAFFE_INIT = {
    "torso": (0.8, 0.9),
    "neck": (2.6, 0.65),
    "head": (0.8, 0.7),
    "fore_upper": (1.1, 0.65),
    "fore_mid": (1.2, 0.6),
    "fore_lower": (1.2, 0.6),
    "fore_foot": (1.0, 0.6),
    "hind_upper": (1.0, 0.75),
    "hind_mid": (1.2, 0.6),
    "hind_lower": (1.2, 0.6),
    "hind_foot": (1.0, 0.6),
    "tail": (1.0, 0.6),
}
# Anatomical limits as (length min, length max, thickness min, thickness max). Without them, overall
# size trades off freely against camera depth and the fit settles on absurd proportions.
GIRAFFE_BOUNDS = {
    "torso": (0.6, 1.1, 0.6, 1.1),
    "neck": (1.5, 4.0, 0.35, 0.9),
    "head": (0.6, 1.2, 0.5, 1.0),
    "fore_upper": (0.8, 1.6, 0.4, 1.0),
    "fore_mid": (0.8, 2.0, 0.35, 1.0),
    "fore_lower": (0.8, 2.0, 0.35, 1.0),
    "fore_foot": (0.7, 1.5, 0.4, 1.0),
    "hind_upper": (0.8, 1.6, 0.4, 1.0),
    "hind_mid": (0.8, 2.0, 0.35, 1.0),
    "hind_lower": (0.8, 2.0, 0.35, 1.0),
    "hind_foot": (0.7, 1.5, 0.4, 1.0),
    "tail": (0.6, 1.6, 0.3, 1.0),
}
# Bones with several children need a stated direction: root -> spine, head -> mouth.
# The spine top (6) branches to neck and both forelegs, so it continues its parent's direction.
_PRIMARY_CHILD = {0: 1, 6: None, 16: 32}


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


class GiraffeBody(nn.Module):
    def __init__(self, v_template, shapedirs, j_regressor, parents, weights, faces, betas, init=None, groups=None, bounds=None):
        super().__init__()
        as_float = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)
        self.register_buffer("v_template", as_float(v_template))
        self.register_buffer("shapedirs", as_float(shapedirs))
        self.register_buffer("j_regressor", as_float(j_regressor))
        self.register_buffer("weights", as_float(weights))
        self.register_buffer("parents", torch.as_tensor(np.asarray(parents).astype(np.int64)))
        self.faces = np.asarray(faces)
        self.register_buffer("mirror", torch.as_tensor(mirror_indices(np.asarray(v_template))))

        groups = BONE_GROUPS if groups is None else groups
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

    def symmetric_offsets(self) -> torch.Tensor:
        mirrored = self.offsets[self.mirror] * torch.tensor([1.0, -1.0, 1.0], device=self.offsets.device)
        return 0.5 * (self.offsets + mirrored)

    def scales(self) -> torch.Tensor:
        """(length, thickness) per bone group, within the anatomical limits."""
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
        vertices = (self.weights[..., None] * moved).sum(dim=1)
        return vertices + self.symmetric_offsets(), new_joints

    def forward(self, global_orient: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        """Posed vertices (N, V, 3) from rotation matrices (N, 3, 3) and (N, J-1, 3, 3)."""
        rest_v, rest_j = self.rest()
        n = global_orient.shape[0]
        rotations = torch.cat([global_orient[:, None], pose], dim=1)
        _, transforms = batch_rigid_transform(rotations, rest_j.expand(n, -1, -1), self.parents)
        blended = (self.weights[None, :, :, None, None] * transforms[:, None]).sum(dim=2)
        homogeneous = torch.cat([rest_v, torch.ones_like(rest_v[:, :1])], dim=1)
        return (blended @ homogeneous[None, :, :, None])[:, :, :3, 0]

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

    def proportions(self) -> dict:
        """Neck/body and leg/body ratios, comparable with `PrimaModel.shape_stats`."""
        _, j = self.rest()
        neck = (j[15] - j[16]).norm()
        body = (j[1] - j[15]).norm()
        leg = (j[7] - j[10]).norm()
        return {"neck_to_body": float(neck / body), "leg_to_body": float(leg / body)}
