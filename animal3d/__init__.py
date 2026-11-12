"""3D body recovery for any quadruped, from images, video or RGB + LiDAR, using PRIMA."""
import os

# Must be set before pyrender is first imported.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
# Mesa's EGL needs /dev/dri access (the `render` group); NVIDIA's needs only /dev/nvidia*.
_NVIDIA_EGL = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
if os.path.exists(_NVIDIA_EGL):
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", _NVIDIA_EGL)
