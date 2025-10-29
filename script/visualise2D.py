import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

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
    "initial_pos": True  ## robot has reached initial position 
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


class LivePosePlotter:
    def __init__(self, maxlen=1000):
        # Store recent values (deque for sliding window)
        self.t = deque(maxlen=maxlen)
        self.pos = [deque(maxlen=maxlen) for _ in range(3)]    # X, Y, Z
        self.euler = [deque(maxlen=maxlen) for _ in range(3)]  # roll, pitch, yaw
        self.start_time = time.time()

        # Setup plots
        self.fig, self.axes = plt.subplots(2, 3, figsize=(12, 6))
        self.lines = []
        for i in range(3):
            # Position subplots
            ax = self.axes[0, i]
            line, = ax.plot([], [], label=f'pos {["X", "Y", "Z"][i]}')
            ax.set_title(f'Position {["X", "Y", "Z"][i]}')
            ax.set_ylim(-2, 2)  # adjust as needed
            ax.set_xlim(0, maxlen)
            self.lines.append(line)

            # Euler subplots
            ax = self.axes[1, i]
            line, = ax.plot([], [], label=f'euler {["roll", "pitch", "yaw"][i]}')
            ax.set_title(f'Euler {["roll", "pitch", "yaw"][i]} (deg)')
            ax.set_ylim(-180, 180)
            ax.set_xlim(0, maxlen)
            self.lines.append(line)
        self.fig.tight_layout()

    def update(self, pos, quat):
        # pos: np.array shape (3,), quat: np.array shape (4,)
        if pos is None or quat is None or len(pos) != 3 or len(quat) != 4:
            return  # Don't update if data is not ready
        t_now = time.time() - self.start_time
        self.t.append(t_now)
        for i in range(3):
            self.pos[i].append(pos[i])

        euler_deg = R.from_quat(quat).as_euler('xyz', degrees=True)
        print(pos, euler_deg)
        for i in range(3):
            self.euler[i].append(euler_deg[i])

    def animate(self, frame):
        t = np.array(self.t)
        for i in range(3):
            y = np.array(self.pos[i])
            n = min(len(t), len(y))
            t_win, y_win = t[-n:], y[-n:]

            # Only set ylim if there is data
            if n > 0:
                max_y = np.max(y_win)
                min_y = np.min(y_win)

                if max_y == 0:
                    ylim = (-1, 1)
                else:
                    ylim = (min_y, 1.1 * max_y)
                self.lines[i].set_data(t_win, y_win)
                self.axes[0, i].set_xlim(max(0, t_win[0]-1), t_win[-1]+1)
                self.axes[0, i].set_ylim(*ylim)

            y_e = np.array(self.euler[i])
            n_e = min(len(t), len(y_e))
            t_e_win, y_e_win = t[-n_e:], y_e[-n_e:]
            if n_e > 0:
                max_ye = np.max(np.abs(y_e_win))
                if max_ye == 0:
                    ylim_e = (-1, 1)
                else:
                    ylim_e = (-1.1 * max_ye, 1.1 * max_ye)
                self.lines[i+3].set_data(t_e_win, y_e_win)
                self.axes[1, i].set_xlim(max(0, t_e_win[0]-1), t_e_win[-1]+1)
                self.axes[1, i].set_ylim(*ylim_e)

            return self.lines

    def show(self):
        self.ani = animation.FuncAnimation(self.fig, self.animate, interval=100, blit=False)
        

def runner(plotter : LivePosePlotter, grasp_pose_node : PoseSubscriber, target_pose_node : PoseSubscriber): 

    while True:
        # --- get current poses ---
        g_dict = grasp_pose_node.get_pose()
        t_dict = target_pose_node.get_pose()

        grasp_pose = make_se3_from_pose_dict(g_dict)   # (p_g, q_g)
        target_pose = make_se3_from_pose_dict(t_dict)  # (p_t, q_t)

        p_rel, q_rel = calc_relative_pose(grasp_pose, target_pose)
        
        plotter.update(p_rel, q_rel)
        time.sleep(0.1)


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

    ## setup live 2d plots
    plotter = LivePosePlotter(maxlen=1000)
    plot_thread = threading.Thread(target=runner, daemon=True, args = [plotter, grasp_pose_node, target_pose_node])
    plot_thread.start()

    plotter.show()
    plt.show()

    # input("Hit enter to close....")
    # Clean shutdown
    executor.shutdown()   # stop spinning threads
    grasp_pose_node.destroy_node()
    target_pose_node.destroy_node()
    rclpy.shutdown()
    return


if __name__ == '__main__':
    main()