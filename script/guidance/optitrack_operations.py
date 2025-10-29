

import numpy as np
from scipy.spatial.transform import Rotation as R

def grasp_pose_in_front_of_target_plusZ(p_target_w, q_target_w, standoff=0.25):
    """
    Build an ABSOLUTE grasp pose so that:
      - The gripper sits 'standoff' meters in FRONT of the target along +Z_target,
      - The gripper looks directly at the target: +Z_gripper = -Z_target,
      - +X_gripper is parallel to +X_target (projected to keep orthonormal).
    Args:
      p_target_w: (3,) target position (world)
      q_target_w: (4,) target quaternion (x,y,z,w)
      standoff  : float meters (e.g., 0.25)
    Returns:
      p_g_w: (3,) desired gripper position (world)
      q_wg : (4,) desired gripper orientation (world->gripper, x,y,z,w)
    """
    R_wt = R.from_quat(np.asarray(q_target_w, float)).as_matrix()
    x_t = R_wt[:, 0]                  # target +X in world
    z_t = R_wt[:, 2]                  # target +Z in world

    # Position: in front of target along +Z_target
    p_g_w = np.asarray(p_target_w, float) + float(standoff) * z_t

    # Orientation: look back at target -> +Z_gripper = -Z_target
    z_g = -z_t

    # Make +X_gripper parallel to +X_target but orthogonal to z_g
    x_g = x_t - z_g * np.dot(x_t, z_g)
    nx = np.linalg.norm(x_g)
    if nx < 1e-9:
        # if x_t is (near) colinear with z_g, fall back to target +Y
        y_t = R_wt[:, 1]
        x_g = y_t - z_g * np.dot(y_t, z_g)
        nx = np.linalg.norm(x_g)
        if nx < 1e-9:
            # extreme degenerate fallback
            x_g = np.array([1.0, 0.0, 0.0]) if abs(z_g[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            x_g = x_g - z_g * np.dot(x_g, z_g)
            nx = np.linalg.norm(x_g)
    x_g /= nx

    # Complete right-handed frame
    y_g = np.cross(z_g, x_g)
    y_g /= (np.linalg.norm(y_g) + 1e-12)
    z_g = np.cross(x_g, y_g)          # re-orthonormalize
    z_g /= (np.linalg.norm(z_g) + 1e-12)

    R_wg = np.column_stack([x_g, y_g, z_g])
    q_wg = R.from_matrix(R_wg).as_quat()

    # Optional sanity check during debugging:
    # p_rel_new = R_wg.T @ (np.asarray(p_target_w) - p_g_w)
    # print("target in NEW grasp frame ->", p_rel_new)  # ~ [0, 0, standoff]

    return p_g_w, q_wg

def reproject_from_relative(pose_grasp, pose_rel):
    """
    Reconstruct target in world from relative pose and current grasp pose.
    """
    p_grasp, q_grasp = pose_grasp
    p_rel, q_rel = pose_rel

    Rg = R.from_quat(q_grasp).as_matrix()
    p_hat = Rg @ p_rel + p_grasp
    R_hat = Rg @ R.from_quat(q_rel).as_matrix()
    q_hat = R.from_matrix(R_hat).as_quat()
    return p_hat, q_hat
