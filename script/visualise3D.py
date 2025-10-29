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
from rclpy.qos import QoSProfile, ReliabilityPolicy

## just helper functions for converting units and bs
from guidance.helpers import *
## drawing and term colouring functions
from guidance.drawing_functions import * 
## the following function can be used to place the grasp in a linear position in front of the target
from guidance.optitrack_operations import grasp_pose_in_front_of_target_plusZ 


'''
## This code needs the optitrack ros node to be running,
## make sure both this machine and the optitrack pc share a network.
## Run the following command in a termnial - the ip address should match the optitrack pc:

ros2 launch vrpn_mocap client.launch.yaml server:=10.212.0.51 port:=3883

## you may need to run the following to check the topics of you camera and target and update them in the code accordingly:

ros2 topic list

'''


# Tolerances
STATE = {
    "break_inner": False, ## to go to next pose
    "break_outer": False, ## to finish the program 
    "initial_pos": False  ## robot has reached initial position 
}


class PoseSubscriber(Node):
    def __init__(self, node_name, topic_name):
        super().__init__(node_name)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        self.subscription = self.create_subscription(
            PoseStamped,
            topic_name,
            self.listener_callback,
            qos)
        
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
    # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    # vis.add_geometry(origin)
    # Red sphere at world origin
    # sphere0 = o3d.geometry.TriangleMesh.create_sphere(radius=0.025)
    # sphere0.paint_uniform_color([1, 0, 0])
    # vis.add_geometry(sphere0)
    # vis.poll_events(); vis.update_renderer()

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

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from collections import deque

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

    # --- get current poses ---
    while True:
        g_dict = grasp_pose_node.get_pose()
        t_dict = target_pose_node.get_pose()
        if g_dict is None or t_dict is None:
            print("[wait] Waiting for OptiTrack poses...")
            time.sleep(0.2)
        else:
            print("OPTITRACK READY")
            break

    grasp_pose = make_se3_from_pose_dict(g_dict)   # (p_g, q_g)
    target_pose = make_se3_from_pose_dict(t_dict)  # (p_t, q_t)

    ## draw the current and next pose - first time 
    h_curr = draw_pose(vis, grasp_pose[0], grasp_pose[1], size=0.12, color=(0,1,0))  # green
    h_trgt = draw_pose(vis, target_pose[0], target_pose[1], size=0.12, color=(0.5,0,0.5))  # Purple for target(Proba)

    ## use the rest of the program in infinite loop
    while not STATE["break_outer"]: 

        # --- get current poses ---
        g_dict = grasp_pose_node.get_pose()
        t_dict = target_pose_node.get_pose()

        grasp_pose = make_se3_from_pose_dict(g_dict)   # (p_g, q_g)
        target_pose = make_se3_from_pose_dict(t_dict)  # (p_t, q_t)

        ## update all the poses 
        update_pose(vis, h_curr, grasp_pose[0], grasp_pose[1])
        update_pose(vis, h_trgt, target_pose[0], target_pose[1])

        vis.poll_events(); vis.update_renderer()
        time.sleep(0.1)


    # input("Hit enter to close....")
    # Clean shutdown
    executor.shutdown()   # stop spinning threads
    grasp_pose_node.destroy_node()
    target_pose_node.destroy_node()
    rclpy.shutdown()
    return


if __name__ == '__main__':
    main()