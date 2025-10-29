
import numpy as np
from scipy.spatial.transform import Rotation as R


def build_T(p_xyz, q_xyzw):
    T = np.eye(4)
    T[:3, :3] = R.from_quat(q_xyzw).as_matrix()
    T[:3, 3] = np.array(p_xyz, dtype=float)
    return T

def decompose_T(T):
    p = T[:3, 3].copy()
    q = R.from_matrix(T[:3, :3]).as_quat()
    return p, q

def make_se3_from_pose_dict(pose_dict):
    p = np.array([pose_dict['x'], pose_dict['y'], pose_dict['z']], dtype=float)
    q = np.array([pose_dict['qx'], pose_dict['qy'], pose_dict['qz'], pose_dict['qw']], dtype=float)
    q = q / np.linalg.norm(q)
    return p, q

def calc_relative_pose(grasp_pose, target_pose):
    """
    Returns (p_rel, q_rel): the target pose expressed in the grasp frame.
    """

    T_g = build_T(*grasp_pose)
    T_t = build_T(*target_pose)

    # target relative to grasp
    T_rel = np.linalg.inv(T_g) @ T_t
    p_rel = T_rel[:3, 3]
    q_rel = R.from_matrix(T_rel[:3, :3]).as_quat()
    return p_rel, q_rel

def compose_body_delta(T_world_body, T_delta_body):
    """
    Apply a delta expressed in the *body (grasp) frame* to a world pose.
    Returns T_world_body_next.
    """
    R_wb = T_world_body[:3, :3]
    t_wb = T_world_body[:3, 3]

    R_delta = T_delta_body[:3, :3]
    t_delta_body = T_delta_body[:3, 3]

    # translate: body-frame vector to world-frame
    t_delta_world = R_wb @ t_delta_body

    T_next = np.eye(4)
    T_next[:3, :3] = R_wb @ R_delta         # right-multiply by body-frame rotation
    T_next[:3, 3]  = t_wb + t_delta_world   # translate in world
    return T_next

def to_euler_deg(q_xyzw):
    # roll (x), pitch (y), yaw (z) in degrees
    return R.from_quat(q_xyzw).as_euler('xyz', degrees=True)
