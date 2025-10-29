import numpy as np
import open3d as o3d
from . import helpers

from termcolor import colored


def color_dim(curr, nxt, tol, fmt="{:.6f}"):
    ok = abs(curr - nxt) <= tol
    txt_curr = colored(fmt.format(curr), "green" if ok else "red")
    txt_next = fmt.format(nxt)
    return f"{txt_curr} : {txt_next}", ok

# ---------- Pose drawing helpers ----------
def draw_pose(vis, pos, quat_xyzw, size=0.10, color=(0, 1, 0), marker_radius=0.01):
    """
    Add a pose (small sphere + coordinate frame) to an existing Visualizer.
    Returns a handle you can later pass to remove_pose() or update_pose().
    """
    T_ = helpers.build_T(pos, quat_xyzw)

    # Small marker
    marker = o3d.geometry.TriangleMesh.create_sphere(radius=marker_radius)
    marker.paint_uniform_color(color)
    marker.transform(T_)

    # Coordinate frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    frame.transform(T_)

    # Add to scene
    vis.add_geometry(marker)
    vis.add_geometry(frame)
    vis.poll_events(); vis.update_renderer()

    # Keep last transform so we can update in-place if you want
    handle = {"marker": marker, "frame": frame, "T": T_.copy()}
    return handle


def remove_pose(vis, handle):
    """Remove previously drawn pose from the Visualizer."""
    for key in ("marker", "frame"):
        vis.remove_geometry(handle[key], reset_bounding_box=False)
    vis.poll_events(); vis.update_renderer()

def update_pose(vis, handle, new_pos, new_quat_xyzw):
    """
    Optional: update an existing pose in-place (faster than remove+add).
    We apply the delta transform: T_delta = T_new * inv(T_old).
    """
    T_old = handle["T"]
    T_new = helpers.build_T(new_pos, new_quat_xyzw)
    T_delta = T_new @ np.linalg.inv(T_old)

    handle["marker"].transform(T_delta)
    handle["frame"].transform(T_delta)
    handle["T"] = T_new.copy()

    vis.update_geometry(handle["marker"])
    vis.update_geometry(handle["frame"])
    vis.poll_events(); vis.update_renderer()