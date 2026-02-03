import torch
import numpy as np
import sys
from pathlib import Path
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

parent_folder = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_folder / "Models"))
sys.path.insert(0, str(parent_folder / "Solvers"))

from CartPole import cartpole_dynamics_batched_torch
from SymplecticRK4_TORCH import symplectic_rk4_step_torch

class CartPoleEnv_srk4_torch(gym.Env):
    
    def __init__(self):
        
        #Parameters
        self.params = (1.0, 0.1, 1.0, 0.01, 9.81)
        self.dt = 0.002
        self.t = 0.0
        self.step_number = 0
        self.max_steps = int(10.0/self.dt)

        #Action space
        self.action_space = gym.spaces.Box(low=-1.0,high=1.0,shape=(1,),dtype=np.float32)

        #Observation space
        self.observation_space = gym.spaces.Box(low = np.array([-3.0,-6*np.pi,-10.0,-20.0]),
                                                high = np.array([3.0,6*np.pi,10.0,20.0]),
                                                shape = (4,),
                                                dtype = np.float32)

        #"Uninitialized" states, will be set with reset()
        self.state = np.zeros([4,1])

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        # State is shape [4], no batch dimension needed here
        self.state = torch.tensor([0.0, np.pi, 0.0, 0.0], dtype=torch.float32)
        self.t = 0.0
        self.step_number = 0
        observation = self.state.numpy()
        info = {}
        return observation, info

    def step(self, action, f_scale=100.0):
        # Convert action to tensor if it's a numpy array
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()
        elif not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)
        
        u = action * f_scale
        
        # Reshape for batched function:
        # state: [4] -> [1, 4] (1 sample with 4 state variables)
        # u: [1] -> [1, 1] (1 sample with 1 action variable)
        state_batched = self.state.unsqueeze(0)  # [4] -> [1, 4]
        u_batched = u.reshape(1, -1)  # Ensure shape [1, num_actions]
        
        # Use symplectic RK4 integrator
        state_next, t_next = symplectic_rk4_step_torch(
            cartpole_dynamics_batched_torch,
            state_batched, 
            self.t, 
            u_batched, 
            self.dt, 
            self.params
        )
        
        # state_next is [1, 4], squeeze to get [4]
        self.state = state_next.squeeze(0)
        self.t = t_next
        
        self.step_number += 1
        
        terminated = False
        if self.step_number >= self.max_steps:
            terminated = True
        
        q1, q2, u1, u2 = self.state
        
        reward = np.cos(q2.item()) - 0.01*q1.item()**2 - 0.001*u1.item()**2 - 0.001*u2.item()**2
        
        observation = self.state.numpy()
        info = {}
        truncated = False
        
        return observation, reward, terminated, truncated, info
        
        
if __name__ == "__main__":
    try:
        check_env(CartPoleEnv_srk4_torch())
        print("Environment passes all checks!")
    except Exception as e:
        print(f"Environment has issues: {e}")
        import traceback
        traceback.print_exc()




