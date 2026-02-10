import torch
import numpy as np
import sys
from pathlib import Path
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import matplotlib.pyplot as plt

parent_folder = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_folder / "Models"))
sys.path.insert(0, str(parent_folder / "Solvers"))

from Acrobot import acrobot_dynamics_batched_torch
from SymplecticRK4_TORCH import symplectic_rk4_step_torch

class AcrobotEnv_srk4_torch(gym.Env):

    def __init__(self):

        self.dynamics_func = torch.compile(acrobot_dynamics_batched_torch)

        self.params = (1.0, 1.0, 1.0/12.0, 9.81)
        self.dt = 0.002
        self.t = 0.0
        self.step_number = 0
        self.max_steps = int(5.0/self.dt)
        self.angle_space = np.deg2rad(5.0)
        self.goal_angle = np.deg2rad(180.0)
        self.low_angle = self.goal_angle - self.angle_space
        self.high_angle = self.goal_angle + self.angle_space

        #Action space
        self.action_space = gym.spaces.Box(low=-1.0,high=1.0,shape=(1,),dtype=np.float32)

        #Observation space
        self.observation_space = gym.spaces.Box(low = np.array([-50.0,-50.0,-50.0,-50.0]),
                                                high = np.array([50.0,50.0,50.0,50.0]),
                                                shape = (4,),
                                                dtype = np.float32)
        
        #"Uninitialized" states, will be set with reset()
        self.state = np.zeros([4,1])

    def reset(self,seed=None,**kwargs):
        super().reset(seed=seed)

        self.state = torch.tensor([np.random.uniform(-np.pi/2, np.pi/2),
                                   np.random.uniform(-np.pi/2, np.pi/2),
                                   np.random.uniform(-5.0,5.0),
                                   np.random.uniform(-5.0,5.0)])


        self.t = 0.0
        self.step_number = 0
        observation = self.state.cpu().numpy()
        info = {}

        return observation, info
    
    def step(self,action,T_scale=200.0):

        # Convert action to tensor if it's a numpy array
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()
        elif not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)
        
        u = action * T_scale

        # Reshape for batched function:
        # state: [4] -> [1, 4] (1 sample with 4 state variables)
        # u: [1] -> [1, 1] (1 sample with 1 action variable)
        state_batched = self.state.unsqueeze(0)  # [4] -> [1, 4]
        u_batched = u.reshape(1, -1)  # Ensure shape [1, num_actions]
        
        # Use symplectic RK4 integrator
        state_next, t_next = symplectic_rk4_step_torch(
            self.dynamics_func,
            state_batched, 
            self.t, 
            u_batched, 
            self.dt, 
            self.params
        )
        
        # state_next is [1, 4], squeeze to get [4]
        self.state = state_next.squeeze(0)


        self.state[2] = torch.clamp(self.state[2], -50.0, 50.0)  # u1
        self.state[3] = torch.clamp(self.state[3], -50.0, 50.0)  # u2


        self.t = t_next
        
        self.step_number += 1
        terminated = False
        q1, q2, u1, u2 = self.state

        ang1 = np.abs(np.arctan2(np.sin(q1.item()), np.cos(q1.item())))
        ang2 = np.abs(np.arctan2(np.sin(q2.item()), np.cos(q2.item())))

        reward = -np.cos(ang1) - np.cos(ang1 + ang2) - 0.01*u1.item()**2 - 0.01*u2.item()**2

        if self.step_number >= self.max_steps:
            terminated = True

        observation = self.state.cpu().numpy()
        info = {}
        truncated = False
        
        return observation, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            self.ax.set_xlim(-2.5, 2.5)
            self.ax.set_ylim(-2.5, 2.5)
            self.ax.set_aspect('equal')
            self.ax.grid(True)
            self.line, = self.ax.plot([], [], 'o-', lw=2, markersize=8)
            plt.ion()
            plt.show()
        
        # Extract angles
        q1, q2 = self.state[0].item(), self.state[1].item()
        
        # Link lengths (assuming both links are length 1.0 based on your params)
        L1, L2 = 1.0, 1.0
        
        # Forward kinematics
        x0, y0 = 0.0, 0.0  # Base
        x1 = L1 * np.sin(q1)
        y1 = -L1 * np.cos(q1)
        x2 = x1 + L2 * np.sin(q1 + q2)
        y2 = y1 - L2 * np.cos(q1 + q2)
        
        # Update plot
        self.line.set_data([x0, x1, x2], [y0, y1, y2])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        """Close the rendering window"""
        if hasattr(self, 'fig'):
            plt.close(self.fig)


if __name__ == "__main__":
    try:
        check_env(AcrobotEnv_srk4_torch())
        print("Environment passes all checks!")
    except Exception as e:
        print(f"Environment has issues: {e}")
        import traceback
        traceback.print_exc()
    