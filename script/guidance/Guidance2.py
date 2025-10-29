import os
import math
import numpy as np
import random as rand
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation as R

from stable_baselines3 import TD3

class AbdullaGuidance():
    def __init__(self, R_pos=None, R_q_sl=None):
        
        print("init: ", R_q_sl)

        ######## Initialise Guidance #####################################################
        ## place model files in the same directory
        script_dir = os.path.dirname(os.path.abspath(__file__)) 
        self.modelR = TD3.load(script_dir + "/TD3_Ren_Working_X23_high_2")
        self.modelA = TD3.load(script_dir + "/TD3_Att_Working_X2_high")
        
        ######## Initialise Physics ######################################################
        self.sample_t = 1
        self.dtime = 0.01
        self.max_episode_time = 400
        
        if R_pos is None:
            self.R_pos = np.array([0, 0, 20])
        else:
            self.R_pos = R_pos
        
        if R_q_sl is None:
            self.R_q_sl = np.array([0, 0, 0, 1])  
        else:
            self.R_q_sl = R_q_sl
        
        self.R_vel = np.array([0, 0, 0])
        self.R_ang_rate = np.array([0, 0, 0])
        
        ######## Initialise Navigation ######################################################
        self.prev_N_ang_err = self._get_ang_err(self.R_q_sl)
        self.prev_N_pos = self.R_pos
        
    def __call__(self, N_pos, N_q_sl):
        
        ######## Create State for Guidance ##################################################
        v_est = (N_pos - self.prev_N_pos) / self.sample_t
        
        N_ang_err = self._get_ang_err(N_q_sl)
        w_est = self._wrap_to_pi(N_ang_err - self.prev_N_ang_err) / self.sample_t
        
        state = np.concatenate((N_pos, v_est, N_ang_err, w_est))
        
        self.prev_N_pos = N_pos
        self.prev_N_ang_err = N_ang_err
        
        actionR = self.modelR.predict(state[:6])
        actionA = self.modelA.predict(state[6:])
        action = np.concatenate((actionR[0], actionA[0]))
        
        ######## Enact Action ###############################################################
        max_control_vel = 0.01 * self.dtime
        max_control_dw = math.pi/72 * self.dtime
        
        control_vel = action[:3] * max_control_vel
        control_dw = action[3:] * max_control_dw
        
        #print("before: ", self.R_q_sl)

        for _ in range(int(self.sample_t/self.dtime)):
            # Propagate Position
            self.R_vel = self.R_vel + control_vel
            self.R_pos = np.array(self.R_pos + self.R_vel*self.dtime)
                        
            # Propagate Attitude
            self.R_ang_rate = self.R_ang_rate + control_dw
            self.R_ang_rate1 = (self.R_ang_rate[0], self.R_ang_rate[1], self.R_ang_rate[2])
            R_q_sf = [self.R_q_sl[3], self.R_q_sl[0], self.R_q_sl[1], self.R_q_sl[2]]

            R_q_sf = solve_ivp(self._derive_q, [0,self.dtime], R_q_sf, args=(self.R_ang_rate1,)).y.T[-1]
            R_q_sf = R_q_sf / np.linalg.norm(R_q_sf)
            
            self.R_q_sl = [R_q_sf[1], R_q_sf[2], R_q_sf[3], R_q_sf[0]]
        
        #print("after: ", self.R_q_sl)

        return {"R_pos": self.R_pos, "R_q_sl": self.R_q_sl}
    
    def _derive_q(self, t, q, w):
        dqdt = 0.5*np.matmul(self._skew_w4(w), q) #q_dot
        return dqdt
    
    def _skew_w4(self, x):
        wx4 = np.array([[  0,  -x[0], -x[1], -x[2]],
                        [x[0],    0,   x[2], -x[1]],
                        [x[1], -x[2],    0,   x[0]],
                        [x[2],  x[1], -x[0],    0]])
        return wx4
    
    def _get_ang_err(self, chs_q_sl):
        chs_q = [chs_q_sl[3], chs_q_sl[0], chs_q_sl[1], chs_q_sl[2]]
        chs_q /= np.linalg.norm(chs_q)
        chs_q = R.from_quat([chs_q[1], chs_q[2], chs_q[3], chs_q[0]])
        ang_err = chs_q.as_euler('xyz', degrees=False)
        return ang_err

    def _wrap_to_pi(self, x):
        return (x + np.pi) % (2*np.pi) - np.pi


if __name__ == "__main__":
    Guidance = AbdullaGuidance()

    angerr, pos = [], []
    N_pos = Guidance.R_pos
    N_q_sl = Guidance.R_q_sl

    for _ in range(500):
        results = Guidance(N_pos, N_q_sl)
        N_pos = results["R_pos"]
        N_q_sl = results["R_q_sl"]
        
        N_ang_err = Guidance._get_ang_err(N_q_sl)
        
        angerr.append(N_ang_err)
        pos.append(N_pos)

    angerr = np.array(angerr)
    pos = np.array(pos)

    # Time vector
    time = np.arange(0, len(pos)*Guidance.sample_t, Guidance.sample_t)
    xstart, xend = 0, len(pos)

    # Position plot
    plt.plot(time, pos[:,0], '-r', label="X")
    plt.plot(time, pos[:,1], '-g', label="Y")
    plt.plot(time, pos[:,2], '-b', label="Z")
    plt.title('Position Error')
    plt.ylabel('Position [m]')
    plt.xlabel('Time [s]')
    plt.xlim([xstart, xend])
    #plt.ylim([-1, 1])
    plt.legend()
    plt.grid()
    plt.show()

    # Angular Error plot
    plt.plot(time, angerr[:,0], '-r', label="roll")
    plt.plot(time, angerr[:,1], '-g', label="pitch")
    plt.plot(time, angerr[:,2], '-b', label="yaw")
    plt.title('Euler angular error')
    plt.ylabel('Euler angles [deg]')
    plt.xlabel('Time [s]')
    plt.xlim([xstart, xend])
    #plt.ylim([-2.5, 2.5])
    plt.legend()
    plt.grid()
    plt.show()


