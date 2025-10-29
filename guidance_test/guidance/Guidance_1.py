import math
import os
import numpy as np
import random as rand
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation as R

from stable_baselines3 import TD3

import gym as gym 
from gym import Env
from gym.spaces import Box

NavInLoop = False

if NavInLoop:
    from navigation import Navigation
    from utils.data_io import dataIO
    import gnc_testLoop as gnc_help

def init_pos():
    err_dist = 0.1
    x = rand.random() * err_dist
    y = rand.random() * err_dist
    z = 20 + rand.random() * err_dist 
    pos = np.array([x, y, z])
    return pos

def randq(max_angerr):
    angs=np.array([rand.random(), rand.random(), rand.random()])*max_angerr
    rot = R.from_euler('xyz', angs, degrees=True)
    q = rot.as_quat()
    q = np.array([q[3], q[0], q[1], q[2]]) 
    return q

def skew_w4(x):
    wx4 = np.array([[  0,  -x[0], -x[1], -x[2]],
                    [x[0],    0,   x[2], -x[1]],
                    [x[1], -x[2],    0,   x[0]],
                    [x[2],  x[1], -x[0],    0]])
    return wx4

def derive_q(self, q, *w):
    dqdt = 0.5*np.matmul(skew_w4(w), q) #q_dot
    return dqdt

def get_ang_err(chs_q):
    chs_q /= np.linalg.norm(chs_q)
    chs_q = R.from_quat([chs_q[1], chs_q[2], chs_q[3], chs_q[0]])
    ang_err = chs_q.as_euler('xyz', degrees=False)
    return ang_err

def wrap_to_pi(x):
    # Wrap angle to (-pi, pi]
    return (x + np.pi) % (2*np.pi) - np.pi

def correct_pose_for_guidance(p_est, q_est):
    # Fix rotation only
    q_fix = np.array([1.0, 0.0, 0.0, 0.0])
    r_fix = R.from_quat(q_fix)
    r_est = R.from_quat(q_est)

    # Only correct orientation
    r_corrected = r_fix * r_est
    q_corrected = r_corrected.as_quat()

    # Keep position as is
    p_corrected = p_est.copy()

    return p_corrected, q_corrected

def NavBlock(posi, chs_q_sl, navi):
    if NavInLoop:
        if posi[2] <= 1.95:
            range_flag = "short"
        else:
            range_flag = "far"

        gnc_help.panguFlight(posi, chs_q_sl, range_flag)
        img = gnc_help.callPangu(range_flag)
        navi.model_status = range_flag
        Npos, Nrot = navi.getPose(img)
        Npos, Nrot = correct_pose_for_guidance(Npos, Nrot)
        # d_q = R.from_quat([1, 0, 0, 0])
        # q_err = d_q.inv() * R.from_quat(Nrot)
    
    else:
        chs_q_sf = [chs_q_sl[3], chs_q_sl[0], chs_q_sl[1], chs_q_sl[2]]
        ang_err = get_ang_err(chs_q_sf)
        
        pos_noise_std = (2.5/100) * np.abs(posi)
        ang_noise_std = (2.5/100) * np.abs(ang_err)
        
        Npos = posi + np.random.normal(0, pos_noise_std, 3)
        Nang_err = ang_err + np.random.normal(0, ang_noise_std, 3)
           
        r = R.from_euler('xyz', Nang_err, degrees=False)
        Nrot = r.as_quat()
        img = None

    return Npos, Nrot, img

sample_t = 1
dtime = 0.01
max_episode_time = 400

max_angerr = 0.25

class FullEnv(Env):
    def __init__(self):
        if NavInLoop:
            self.experimentSavePath = "experiment_results"
            self.experiment_name = "gnc_test_13082025"
            
            ## navigation
            self.navi = Navigation(start_range="far")
            self.result_io = dataIO(self.experiment_name, self.experimentSavePath)
        else:
            self.navi = None
            
        self.counter = 0
    
        # Setup Env
        self.action_space = Box(-1, 1,shape=(6,),dtype=np.float32)
        self.observation_space = Box(-np.inf,np.inf,shape=(12,),dtype=np.float32)
        
        # Initialise Position
        self.pos  = np.array([0, 0, 20])
        self.vel = np.array([0, 0, 0])

        # Initialise Attitude 
        self.chs_q = np.array([1, 0, 0, 0])     
        self.chs_w = np.array([0, 0, 0])
        
        # Pass to Navigation Block (add noise)
        chs_q_sl = [self.chs_q[1], self.chs_q[2], self.chs_q[3], self.chs_q[0]] #make scalar last
        self.Npos = self.pos
        Nchs_q_sl = chs_q_sl
        self.Nchs_q = [Nchs_q_sl[3], Nchs_q_sl[0], Nchs_q_sl[1], Nchs_q_sl[2]] #make scalar first
        self.Nang_err = get_ang_err(self.Nchs_q) #find ang error from quat
        
        # Initialise State
        self.state = np.concatenate((self.Npos, self.vel, self.Nang_err, self.chs_w))
        self.orbit_time = max_episode_time

        
    def step(self, action):
        self.counter += 1
        
        if NavInLoop:
            nav_status = self.navi.model_status
        
        # store prev state
        prev_Npos = self.Npos.copy()
        prev_Nang_err = self.Nang_err.copy()
        prev_Nchs_q = np.array(self.Nchs_q)
        
        # Setup Action
        max_control_vel = 0.01 * dtime #max is 1 cm/s2
        max_control_dw = math.pi/72 * dtime #max is 2.5 deg/s2
        self.control_vel = action[:3] * max_control_vel
        self.control_dw = action[3:] * max_control_dw
        for _ in range(int(sample_t/dtime)):
            # Propagate Position
            self.vel = self.vel + self.control_vel
            self.pos = np.array(self.pos + self.vel*dtime)
            
            # Propagate Attitude
            self.chs_w = self.chs_w + self.control_dw
            self.chs_w1 = (self.chs_w[0], self.chs_w[1], self.chs_w[2])
            self.chs_q = solve_ivp(derive_q, [0,dtime], self.chs_q, args=self.chs_w1).y.T[-1]
            
        # Pass to Navigation Block (add noise)
        chs_q_sl = [self.chs_q[1], self.chs_q[2], self.chs_q[3], self.chs_q[0]] #make scalar last
        #print("###############################################################################################################")
        self.Npos, Nchs_q_sl, img = NavBlock(self.pos, chs_q_sl, self.navi) #send to nav block
        
        if NavInLoop:
            if (nav_status == "far") and (self.navi.model_status=="short"):
                self.result_io.saveCSV(gncSaveName = "gnc_results_far.csv", navSaveName = "nav_results_far.csv")
            
            self.result_io.insertResult(self.counter, img, self.pos, chs_q_sl, self.Npos, Nchs_q_sl)
        
        #print("Pose GNC: ", self.counter, "GNC pos: ", self.pos, "GNC rot: ", R.from_quat(chs_q_sl).as_euler('xyz', degrees=True))
        #print("Pose NAV: ", self.counter, "NAV pos: ", self.Npos, "NAV rot: ", R.from_quat(Nchs_q_sl).as_euler('xyz', degrees=True))

        self.Nchs_q = [Nchs_q_sl[3], Nchs_q_sl[0], Nchs_q_sl[1], Nchs_q_sl[2]] #make scalar first
        self.Nang_err = get_ang_err(self.Nchs_q) #find ang error from quat
                
        # Propagate State
        v_est = (self.Npos - prev_Npos) / sample_t
        dang = wrap_to_pi(self.Nang_err - prev_Nang_err)
        w_est = dang / sample_t
        
        self.state = np.concatenate((self.Npos, v_est, self.Nang_err, w_est))
        self.orbit_time -= sample_t
        done = (self.orbit_time <= 0) or (np.linalg.norm(self.pos) >= 25) or (np.linalg.norm(self.chs_w) >= math.pi/18) or (np.linalg.norm(self.pos) <= 0.1) or (self.pos[2] <= 0.08) #or (self.navi.model_status == "done") 
        info = {"state": self.state, "action": action, "dist": np.linalg.norm(self.pos)}
                
        return self.state, 0, done, info
        
    def reset(self):
        
        # Initialise Position
        #self.pos = init_pos()
        self.pos  = np.array([0.1, 0.1, 20])
        self.vel = np.array([0, 0, 0])
        self.counter = 0
        print(self.pos)
        # Initialise Attitude
        #self.chs_q = randq(max_angerr)  
        self.chs_q = np.array([1, 0, 0, 0])     
        self.chs_w = np.array([0, 0, 0])
        
        # Pass to Navigation Block (add noise)
        chs_q_sl = [self.chs_q[1], self.chs_q[2], self.chs_q[3], self.chs_q[0]] #make scalar last
        self.Npos, Nchs_q_sl, img = NavBlock(self.pos, chs_q_sl, self.navi) #send to nav block
        print("step: ", self.counter, "GNC pos: ", self.pos, "GNC: ", R.from_quat(chs_q_sl).as_euler('xyz', degrees=True))
        print("step: ", self.counter, "NAV pos: ", self.Npos, "NAV rot: ", R.from_quat(Nchs_q_sl).as_euler('xyz', degrees=True))
        
        if NavInLoop:
            self.result_io.insertResult(self.counter, img, self.pos, chs_q_sl, self.Npos, Nchs_q_sl)
        
        self.Nchs_q = [Nchs_q_sl[3], Nchs_q_sl[0], Nchs_q_sl[1], Nchs_q_sl[2]] #make scalar first
        self.Nang_err = get_ang_err(self.Nchs_q) #find ang error from quat
                
        # Initialise State
        self.state = np.concatenate((self.Npos, self.vel, self.Nang_err, self.chs_w))
        self.orbit_time = max_episode_time
        
        return self.state
    
env = FullEnv()

modelR = TD3.load("./TD3_Ren_Working_X20_high_1.zip")
modelA = TD3.load("./TD3_Att_Working_X1_high.zip")

state = env.reset()
done = False
infos = []

while not done:
    actionR = modelR.predict(state[:6])
    actionA = modelA.predict(state[6:])
    action = np.concatenate((actionR[0], actionA[0]))
    
    next_state, _, done, info = env.step(action)
    infos.append(info)
    state = next_state

if NavInLoop:
    env.result_io.saveCSV()
    drl_plot_path = os.path.join(env.experiment_name, env.experimentSavePath)

pos, vel, dist, ActR = [], [], [], []
ang, dang, ActA  = [], [], []

pos.append(np.array([info['state'][:3] for info in infos]))
vel.append(np.array([info['state'][3:6] for info in infos]))
dist.append(np.array([info['dist'] for info in infos]))
ActR.append(np.array([info['action'][:3] for info in infos]))

ang.append(np.array([info['state'][6:9]*(180/math.pi) for info in infos]))
dang.append(np.array([info['state'][9:]*(180/math.pi) for info in infos]))
ActA.append(np.array([info['action'][3:] for info in infos]))

# Time vector
time = np.arange(0, len(pos[0])*sample_t, sample_t)
xstart, xend = 0, len(pos[0])

# Position plot
plt.plot(time, pos[0][:,0], '-r', label="X")
plt.plot(time, pos[0][:,1], '-g', label="Y")
plt.plot(time, pos[0][:,2], '-b', label="Z")
plt.plot(time, dist[0], 'k-', label="Distance", linewidth=2)
plt.title('Position Error')
plt.ylabel('Position [m]')
plt.xlabel('Time [s]')
plt.xlim([xstart, xend])
#plt.ylim([-3, 1])
plt.legend()
plt.grid()
plt.show()
#plt.savefig(os.path.join(drl_plot_path,'DRL Position.png'))

# Velocity plot
plt.plot(time, vel[0][:,0], '-r', label="dX")
plt.plot(time, vel[0][:,1], '-g', label="dY")
plt.plot(time, vel[0][:,2], '-b', label="dZ")
plt.title('Relative Velocity')
plt.ylabel('Velocity [m/s]')
plt.xlabel('Time [s]')
plt.xlim([xstart, xend])
#plt.ylim([-0.08, 0.08])
plt.legend()
plt.grid()
plt.show()
#plt.savefig(os.path.join(drl_plot_path,'DRL Relative Velocity.png'))

# Angular Error plot
plt.plot(time, ang[0][:,0], '-r', label="roll")
plt.plot(time, ang[0][:,1], '-g', label="pitch")
plt.plot(time, ang[0][:,2], '-b', label="yaw")
plt.title('Euler angular error')
plt.ylabel('Euler angles [deg]')
plt.xlabel('Time [s]')
plt.xlim([xstart, xend])
#plt.ylim([-2.5, 2.5])
plt.legend()
plt.grid()
plt.show()
#plt.savefig(os.path.join(drl_plot_path,'DRL Angular Error.png'))

# Angular Velocity plot
plt.plot(time, dang[0][:,0], '-r', label="droll")
plt.plot(time, dang[0][:,1], '-g', label="dpitch")
plt.plot(time, dang[0][:,2], '-b', label="dyaw")
plt.title('Angular rates')
plt.ylabel('Angular rate [deg/s]')
plt.xlabel('Time [s]')
plt.xlim([xstart, xend])
#plt.ylim([-0.08, 0.08])
plt.legend()
plt.grid()
plt.show()
#plt.savefig(os.path.join(drl_plot_path,'DRL Angular Velocity.png'))

# Control/Action plot (Ren)
plt.plot(time, ActR[0][:,0], '-r', label="dVx")
plt.plot(time, ActR[0][:,1], '-g', label="dVy")
plt.plot(time, ActR[0][:,2], '-b', label="dVz")
plt.title('Control Velocity')
plt.ylabel('dVel [m/s]')
plt.xlabel('Time [s]')
plt.xlim([xstart, xend])
plt.legend()
plt.grid()
plt.show()
#plt.savefig(os.path.join(drl_plot_path,'DRL Control Action (Ren).png'))

# Control/Action plot (Att)
plt.plot(time, ActA[0][:,0], '-r', label="dVx")
plt.plot(time, ActA[0][:,1], '-g', label="dVy")
plt.plot(time, ActA[0][:,2], '-b', label="dVz")
plt.title('Control Angular Velocity')
plt.ylabel('dAngVel [deg/s]')
plt.xlabel('Time [s]')
plt.xlim([xstart, xend])
plt.legend()
plt.grid()
plt.show()
#plt.savefig(os.path.join(drl_plot_path,'DRL Control Action (Att).png'))
