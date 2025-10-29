import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import numpy as np
import threading
import time
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import os
import pandas as pd
import open3d as o3d

## just helper functions for converting units and bs
from guidance.helpers import *
## drawing and term colouring functions
from guidance.drawing_functions import * 
## the following function can be used to place the grasp in a linear position in front of the target
from guidance.optitrack_operations import grasp_pose_in_front_of_target_plusZ 

## import class responsible of Abdulla's guidance 
from guidance.Guidance2 import AbdullaGuidance

from termcolor import colored

'''
## This code needs the optitrack ros node to be running,
## make sure both this machine and the optitrack pc share a network.
## Run the following command in a termnial - the ip address should match the optitrack pc:

ros2 launch vrpn_mocap client.launch.yaml server:=10.212.0.51 port:=3883

## you may need to run the following to check the topics of you camera and target and update them in the code accordingly:

ros2 topic list

'''

## path to save data for debugging
data_logger_path = "2025080613_test_1"

# Tolerances
POS_TOL = 1e-3            # 1 mm
EUL_TOL_DEG = .1         # 0.1 degree
STATE = {
    "break_inner": False, ## to go to next pose
    "break_outer": False, ## to finish the program 
    "initial_pos": False  ## robot has reached initial position 
    }

final_pos_ = np.array([0.009986284406087,-0.020930698474099,0.064042380482062])/2
print(final_pos_)
final_rot_ = np.array([0.00243449634766, 0.003991599520204, 0.00316104665867,-0.999984073945593])

print(R.from_quat(final_rot_).as_euler("xyz", degrees=True))

class PoseSubscriber(Node):
    def __init__(self, node_name, topic_name):
        super().__init__(node_name)
        self.subscription = self.create_subscription(
            PoseStamped,
            topic_name,
            self.listener_callback,
            10)
        
        self.pos_digits=6
        self.quat_digits=8

        self.latest_pose = None
        self.pose_lock = threading.Lock()

    def listener_callback(self, msg):
        with self.pose_lock:
            self.latest_pose = {
                'qw': msg.pose.orientation.w,
                'qx': msg.pose.orientation.x,
                'qy': msg.pose.orientation.y,
                'qz': msg.pose.orientation.z,
                'x': msg.pose.position.x,
                'y': msg.pose.position.y,
                'z': msg.pose.position.z
            }
    
    def get_pose(self):
        with self.pose_lock:
            if self.latest_pose is None:
                return None 
            else:
                return self.round_pose(self.latest_pose.copy())
            
    def round_pose(self, pose_dict):
        return {
            k: np.float32(round(v, self.quat_digits if k.startswith('q') else self.pos_digits))
            for k, v in pose_dict.items()
        }


######################################################################################################

def combine_pose_dicts(*pose_dicts):
    """
    Combines multiple pose dictionaries into a single dictionary.
    Accepts any number of dictionaries and merges them.
    Later keys will override earlier ones if duplicated.
    """
    combined = {}
    for d in pose_dicts:
        combined.update(d)
    return combined

def pose_to_dict(pose, prefix):
    """
    Converts a pose tuple into a dictionary with prefixed keys.
    pose: (position, quaternion) where each is a list or np array
    prefix: string prefix for each key (e.g., 'grasp')
    """
    pos, quat = pose
    pos = np.array(pos).flatten()
    quat = np.array(quat).flatten()

    keys = [f"{prefix}_x", f"{prefix}_y", f"{prefix}_z",
            f"{prefix}_qx", f"{prefix}_qy", f"{prefix}_qz", f"{prefix}_qw"]
    values = list(pos) + list(quat)
    return dict(zip(keys, values))


# ---------- Visualizer lifecycle ----------
def make_vis(
        window_name="Pose Viewer",
        width=1280,
        height=720,
        break_inner_key="N",
        break_outer_key="Q"
        ):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=window_name, width=width, height=height)
    # Global origin
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    vis.add_geometry(origin)
    # Red sphere at world origin
    sphere0 = o3d.geometry.TriangleMesh.create_sphere(radius=0.025)
    sphere0.paint_uniform_color([1, 0, 0])
    vis.add_geometry(sphere0)
    vis.poll_events(); vis.update_renderer()

    # --- tiny helper so we can register with char keys or raw codes
    def _register(key, cb):
        code = key if isinstance(key, int) else ord(str(key).upper())
        vis.register_key_callback(code, cb)

    # --- callbacks
    def _cb_break_inner(_vis):
        STATE["break_inner"] = True
        print(f"[key] '{break_inner_key}' pressed → leaving INNER loop after this iteration.")
        return False  # keep visualizer running

    def _cb_break_outer(_vis):
        STATE["break_outer"] = True
        STATE["break_inner"] = True ## need to break the inner loop also to propperly leave 
        print(f"[key] '{break_outer_key}' pressed → leaving OUTER loop.")
        return False

    # --- register keys
    _register(break_inner_key, _cb_break_inner)   # 'N' for next pose
    _register(break_outer_key, _cb_break_outer)   # 'Q' to quit 
    return vis


def to_guidance(p_GW, q_GW, p_TW, q_TW, rotate_x_180=True):
    
    # This is so that the grasp faces the target at 0deg err around x (needed for guidance)
    if rotate_x_180: 
        x_rot = R.from_matrix([
            [-1, 0, 0],
            [ 0, 1, 0],
            [ 0, 0, 1]
        ])
        q_GW = (x_rot * R.from_quat(q_GW)).as_quat()
        
    # Convert target quaternion to Rotation object (scalar-last assumed)
    R_TW = R.from_quat(q_TW)       # target frame wrt world
    R_WT = R_TW.inv()                    # world wrt target
    # Relative position in world
    p_GTW = p_GW - p_TW
    # Transform relative position into target frame
    p_GT = R_WT.apply(p_GTW)
    # Convert chaser and target quaternions to Rotation objects
    R_GW = R.from_quat(q_GW)
    R_GT = R_WT * R_GW                  # chaser w.r.t. target
    # Convert to scalar-last quaternion
    q_GT = R_GT.as_quat()  # [x, y, z, w] (scalar-last)
    return p_GT, q_GT

def from_guidance(p_GT, q_GT, p_TW, q_TW, rotate_x_180=True):
    # Rotation from target to world
    R_TW = R.from_quat(q_TW)
    # Transform relative position back to world frame
    p_GW = p_TW + R_TW.apply(p_GT)
    # Get target and relative orientations
    R_TW = R.from_quat(q_TW)
    R_GT = R.from_quat(q_GT)
    # Chaser orientation in world frame
    R_GW = R_TW * R_GT
    
    # This is because the grasp should be facing the target at 0deg err on x
    if rotate_x_180: 
        x_rot = R.from_matrix([
            [-1,  0,  0],
            [ 0,  1,  0],
            [ 0,  0,  1]
        ])
        R_GW = x_rot * R_GW

    q_GW = R_GW.as_quat()  # scalar-last

    return p_GW, q_GW 

def call_navigation():
    p_GT = [0, 0, 0]
    q_GT = [0, 0, 0, 1]
    return p_GT, q_GT


""" Frames + Conventions
G: Grasp frame 
T: Target frame
W: World frame (OptiTrack frame)

q_GW = Quaternion of grasp in world frame
R_TW = Rotation of target in world frame
p_GT = Position of grasp in target frame
"""

def main():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("This scrip is in:", script_dir)

    # Ensure the logger path is saved exists
    Path(data_logger_path).mkdir(parents=True, exist_ok=True)

    rclpy.init()

    ## setup nodes to get the current position and rotation of the camera and the target objects from optitrack
    grasp_pose_node = PoseSubscriber('grasp_pose_listener', '/vrpn_mocap/Grasp/pose')
    target_pose_node = PoseSubscriber('target_pose_listener', '/vrpn_mocap/Proba/pose')

    # Spin node in a separate thread
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(grasp_pose_node)
    executor.add_node(target_pose_node)

    # Spin in a separate thread
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    ## create visualiser 
    vis = make_vis()
    results = None

    data_logging = []

    optitracking = False #if you want to feed optitrack into guidance
    navigate = False #if you want to feed navigation into guidance 
    # if you turn them both off then guidance feeds into itself

    ## use the rest of the program in infinite loop
    while not STATE["break_outer"]: 

        # reset inner flag for this target/step
        STATE["break_inner"] = False

        # --- get current poses ---
        g_dict = grasp_pose_node.get_pose()
        t_dict = target_pose_node.get_pose()
        if g_dict is None or t_dict is None:
            print("[wait] Waiting for OptiTrack poses...")
            time.sleep(0.2)
            continue

        grasp_pose = make_se3_from_pose_dict(g_dict)   # (p_g, q_g)
        target_pose = make_se3_from_pose_dict(t_dict)  # (p_t, q_t)q

        # Get the relative positions using optitrack (pq_GT, q_GT)
        # Or the navigation
        if navigate:
            rel_pos, rel_quat_xyzw = call_navigation()
        else:
            if optitracking:
                rel_pos, rel_quat_xyzw = to_guidance(*grasp_pose, *target_pose)
            else:
                rel_pos, rel_quat_xyzw  = [0, 0, 5], [0, 0, 0, 1] 
                
        if not STATE["initial_pos"]:
            ## get the robot to a position to start 
            ## use the absolute position of the target to align the grasp in front of it
            grasp_p_next, grasp_q_next = grasp_pose_in_front_of_target_plusZ(*target_pose, standoff=5)

            # grasp_p_next, grasp_q_next = from_guidance(final_pos_, final_rot_,target_pose[0], target_pose[1], rotate_x_180=True)

            ## draw the current and next pose - first time 
            h_curr = draw_pose(vis, grasp_pose[0], grasp_pose[1], size=0.12, color=(0,1,0))  # green
            h_next = draw_pose(vis, grasp_p_next, grasp_q_next, size=0.12, color=(0,0,1))  # blue ## target for robot 
            h_trgt = draw_pose(vis, target_pose[0], target_pose[1], size=0.12, color=(0.5,0,0.5))  # Purple for target(Proba)

            ## define abdulla's guidance block:
            abdulla_g = AbdullaGuidance(rel_pos, rel_quat_xyzw)

            STATE["initial_pos"] = True
            
        else:
            if not optitracking and results is not None:
                if navigate:
                    rel_pos, rel_quat_xyzw = call_navigation()
                else:
                    rel_pos, rel_quat_xyzw = results["R_pos"], results["R_q_sl"]

            ## use abdulla's guidance to get to next postion 
            if type(rel_pos) is not np.ndarray: 
                print(type(rel_pos), type(rel_quat_xyzw))
                rel_pos = np.array(rel_pos)
                rel_quat_xyzw = np.array(rel_quat_xyzw)
            results = abdulla_g(rel_pos, rel_quat_xyzw)
            grasp_p_next, grasp_q_next = from_guidance(results["R_pos"], results["R_q_sl"], *target_pose)
            
            print("Grasp pose   :",grasp_pose)
            print("Next pose    :",grasp_p_next, grasp_q_next)

            ## update all the poses 
            update_pose(vis, h_curr, grasp_pose[0], grasp_pose[1])
            update_pose(vis, h_next, grasp_p_next, grasp_q_next)
            update_pose(vis, h_trgt, target_pose[0], target_pose[1])

            while not STATE["break_inner"] :
                vis.poll_events(); vis.update_renderer()
                time.sleep(0.1)
                pass

            STATE["break_inner"] = False
        
        ## we need read the current postion and print out the T_grasp_next
        while not STATE["break_inner"] :
            ## get your current location 
            g_now = grasp_pose_node.get_pose()

            p_curr = np.array([g_now['x'], g_now['y'], g_now['z']], dtype=float)
            q_curr = np.array([g_now['qx'], g_now['qy'], g_now['qz'], g_now['qw']], dtype=float)

            # Euler angles (deg)
            e_curr = to_euler_deg(q_curr)
            e_next = to_euler_deg(grasp_q_next)

            # Print (per-dimension color for CURRENT value)
            x_line, ok_x = color_dim(p_curr[0], grasp_p_next[0], POS_TOL, "{:.6f}")
            y_line, ok_y = color_dim(p_curr[1], grasp_p_next[1], POS_TOL, "{:.6f}")
            z_line, ok_z = color_dim(p_curr[2], grasp_p_next[2], POS_TOL, "{:.6f}")

            ex_line, ok_ex = color_dim(e_curr[0], e_next[0], EUL_TOL_DEG, "{:.3f}")
            ey_line, ok_ey = color_dim(e_curr[1], e_next[1], EUL_TOL_DEG, "{:.3f}")
            ez_line, ok_ez = color_dim(e_curr[2], e_next[2], EUL_TOL_DEG, "{:.3f}")

            ## update the current pose
            update_pose(vis, h_curr, p_curr, q_curr)

            print(f"x_current : x_next  -> {x_line}")
            print(f"y_current : y_next  -> {y_line}")
            print(f"z_current : z_next  -> {z_line}")
            print(f"euler_x_current : euler_x_next -> {ex_line}")
            print(f"euler_y_current : euler_y_next -> {ey_line}")
            print(f"euler_z_current : euler_z_next -> {ez_line}")
            print("-" * 60)

            # Pace the console a bit
            time.sleep(0.1)
            # break
        
        if results is not None:
            # p_res_g, q_res_g, p_disp, q_disp = 0, 0, 0, 0
            grasp_pose_dict = pose_to_dict(grasp_pose, "grasp")
            target_pose_dict = pose_to_dict(target_pose, "target")
            rel_pose_dict = pose_to_dict((rel_pos, rel_quat_xyzw), "relative")
            # guidance_pose_dict = pose_to_dict((p_res_g, q_res_g), "guidance_out")
            # disp_pose_dict = pose_to_dict((p_disp, q_disp), "disp_out")
            next_pose_dict = pose_to_dict((grasp_p_next, grasp_q_next), "next_out")

            data_logging.append(combine_pose_dicts(grasp_pose_dict, target_pose_dict, rel_pose_dict, next_pose_dict))

        # If we’re here, success == True for this target: shut down and exit.
        if not STATE["break_outer"]:
            print(colored("Moving to next pose.", "yellow"))

    df = pd.DataFrame(data_logging)
    df.to_csv("poses.csv", index=False)

    # input("Hit enter to close....")
    # Clean shutdown
    executor.shutdown()   # stop spinning threads
    grasp_pose_node.destroy_node()
    target_pose_node.destroy_node()
    rclpy.shutdown()
    return


if __name__ == '__main__':
    main()