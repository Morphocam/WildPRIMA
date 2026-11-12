"""Offscreen mesh rendering: full-frame overlays, per-animal silhouettes and fixed-scale profiles."""
from __future__ import annotations

import numpy as np
import pyrender
import trimesh

# Categorical slots from the reference palette; the on-screen ID label also carries identity.
_TRACK_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")


def track_color(track_id: int) -> tuple[float, float, float]:
    """RGB in 0..1. IDs beyond the palette reuse its hues; labels keep them distinguishable."""
    value = _TRACK_COLORS[(track_id - 1) % len(_TRACK_COLORS)].lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def camera_space(vertices: np.ndarray, cam_t: np.ndarray) -> np.ndarray:
    """PRIMA vertices + translation (x right, y down, z forward) to OpenGL (y up, z backward)."""
    return (vertices + cam_t) * np.array([1.0, -1.0, -1.0], dtype=vertices.dtype)


def _pyrender_mesh(vertices: np.ndarray, faces: np.ndarray, color) -> pyrender.Mesh:
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0, roughnessFactor=0.7, alphaMode="OPAQUE", baseColorFactor=(*color, 1.0)
    )
    return pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices, faces, process=False), material=material, smooth=True)


def _add_lights(scene: pyrender.Scene, intensity: float = 2.0) -> None:
    for yaw, pitch in ((0.0, -0.5), (2.1, -0.5), (4.2, -0.5)):
        pose = trimesh.transformations.euler_matrix(pitch, yaw, 0.0)
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=intensity), pose=pose)


class FrameRenderer:
    """Renders into the video frame's pinhole camera, so overlays align with the input pixels.

    Renderers here live as long as the process: deleting one tears down the EGL display it shares
    with any other, which then crashes.
    """

    def __init__(self, width: int, height: int, camera, faces: np.ndarray):
        """`camera` has fx, fy, cx, cy in pixels."""
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.camera = pyrender.IntrinsicsCamera(fx=camera.fx, fy=camera.fy, cx=camera.cx, cy=camera.cy, zfar=1e6)
        self.intrinsics = camera
        self.faces = faces

    def silhouette(self, vertices: np.ndarray, cam_t: np.ndarray) -> np.ndarray:
        scene = pyrender.Scene(ambient_light=np.ones(3))
        scene.add(_pyrender_mesh(camera_space(vertices, cam_t), self.faces, (1.0, 1.0, 1.0)))
        scene.add(self.camera)
        return self.renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY) > 0

    def depth(self, vertices: np.ndarray, cam_t: np.ndarray) -> np.ndarray:
        """Camera-space depth per pixel (0 where the mesh is absent)."""
        scene = pyrender.Scene(ambient_light=np.ones(3))
        scene.add(_pyrender_mesh(camera_space(vertices, cam_t), self.faces, (1.0, 1.0, 1.0)))
        scene.add(self.camera)
        return self.renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)

    def overlay(self, frame_rgb: np.ndarray, vertices, cam_ts, colors, alpha: float = 0.8) -> np.ndarray:
        """Composite all meshes in one scene, so animals occlude one another correctly."""
        if len(vertices) == 0:
            return frame_rgb.copy()
        scene = pyrender.Scene(ambient_light=(0.35, 0.35, 0.35), bg_color=(0, 0, 0, 0))
        for verts, cam_t, color in zip(vertices, cam_ts, colors):
            scene.add(_pyrender_mesh(camera_space(verts, cam_t), self.faces, color))
        scene.add(self.camera)
        _add_lights(scene)
        color, depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        mask = (depth > 0)[..., None] * alpha
        return (frame_rgb * (1 - mask) + color[..., :3] * mask).astype(np.uint8)



class ProfileRenderer:
    """Each animal in its own body frame at one fixed scale.

    Global rotation is removed, so this always shows the side of the animal, and because the scale
    never changes, a change in drawn size means PRIMA changed the *predicted shape*, not the camera.
    """

    def __init__(self, size: int, faces: np.ndarray, half_extent: float, view: np.ndarray):
        self.renderer = pyrender.OffscreenRenderer(size, size)
        self.camera = pyrender.OrthographicCamera(xmag=half_extent, ymag=half_extent, znear=0.01, zfar=100.0)
        self.half_extent = half_extent
        self.view = view  # 3x3: body frame -> OpenGL view axes (x right, y up, z toward viewer)
        self.faces = faces

    def __call__(self, vertices: np.ndarray, global_orient: np.ndarray, color) -> np.ndarray:
        body = vertices @ global_orient  # row-vector form of R^T v
        view = body @ self.view.T
        # Centre horizontally, and rest the lowest point on a fixed ground line.
        offset = np.array([-(view[:, 0].min() + view[:, 0].max()) / 2, -self.half_extent * 0.9 - view[:, 1].min(), -10.0])
        scene = pyrender.Scene(ambient_light=(0.35, 0.35, 0.35), bg_color=(0.97, 0.97, 0.96, 1.0))
        scene.add(_pyrender_mesh(view + offset, self.faces, color))
        scene.add(self.camera)
        _add_lights(scene)
        rendered, _ = self.renderer.render(scene)
        return rendered



def export_mesh(path, vertices: np.ndarray, cam_t: np.ndarray, faces: np.ndarray, color) -> None:
    """Write an OBJ in camera coordinates, keeping each animal's placement in the scene."""
    rgba = np.array([*color, 1.0])
    mesh = trimesh.Trimesh(camera_space(vertices, cam_t), faces, vertex_colors=np.tile(rgba, (len(vertices), 1)), process=False)
    mesh.export(path)


def turntable_frames(vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray, size: int = 720, count: int = 72):
    """Yields RGB frames of a colored mesh (x forward, z up) making one full turn about its up axis."""
    renderer = pyrender.OffscreenRenderer(size, size)
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    half = float(np.linalg.norm(vertices - center, axis=1).max()) * 1.05
    camera = pyrender.OrthographicCamera(xmag=half, ymag=half, znear=0.01, zfar=100.0)
    rgba = np.c_[colors, np.ones(len(colors))]
    for i in range(count):
        yaw = 2 * np.pi * i / count
        c, s = np.cos(yaw), np.sin(yaw)
        turned = (vertices - center) @ np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
        view = np.c_[turned[:, 0], turned[:, 2], -turned[:, 1]] - np.array([0.0, 0.0, 10.0])
        mesh = trimesh.Trimesh(view, faces, vertex_colors=rgba, process=False)
        scene = pyrender.Scene(ambient_light=(0.75, 0.75, 0.75), bg_color=(0.97, 0.97, 0.96, 1.0))
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
        scene.add(camera)
        _add_lights(scene, 0.8)
        yield renderer.render(scene)[0]
