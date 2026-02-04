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

from CartPole import cartpole_dynamics_batched_torch
from SymplecticRK4_TORCH import symplectic_rk4_step_torch

class CartPoleEnv_srk4_torch(gym.Env):
    
    def __init__(self):
        
        #Parameters

        self.dynamics_func = torch.compile(cartpole_dynamics_batched_torch)


        self.params = (1.0, 0.1, 1.0, 0.01, 9.81)
        self.dt = 0.002
        self.t = 0.0
        self.step_number = 0
        self.max_steps = int(5.0/self.dt)
        self.max_angle = np.pi/3
        self.max_dist = 3.0

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
        self.state = torch.tensor([np.random.uniform(-0.5,0.5),
                                np.random.uniform(-np.pi/6, np.pi/6),
                                np.random.uniform(-0.5,0.5),
                                np.random.uniform(-0.5,0.5),], 
                                dtype=torch.float32)
        self.t = 0.0
        self.step_number = 0
        observation = self.state.cpu().numpy()
        info = {}
        return observation, info

    def step(self, action, f_scale=300.0):
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
            self.dynamics_func,
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
        q1, q2, u1, u2 = self.state

        angle = np.arctan2(np.sin(q2.item()), np.cos(q2.item()))

        if np.abs(q1.item()) > self.max_dist or np.abs(q2.item()) > self.max_angle:
            terminated = True
    
        if terminated:
            reward = 0
        else:
            reward = 1 + np.cos(angle) - 0.1*u1.item()**2


        if self.step_number >= self.max_steps:
            terminated = True
            reward = 100
        
        observation = self.state.cpu().numpy()
        info = {}
        truncated = False
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the cart-pole system using matplotlib"""
        if not hasattr(self, 'fig'):
            # Initialize plot on first call
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 6))
            self.ax.set_xlim(-3, 3)
            self.ax.set_ylim(-2, 2)
            self.ax.set_aspect('equal')
            self.ax.grid(True)
            
        # Clear previous frame
        self.ax.clear()
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-2, 2)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        
        # Extract state
        q1, q2, u1, u2 = self.state
        cart_x = q1.item() if isinstance(q1, torch.Tensor) else q1
        pole_angle = q2.item() if isinstance(q2, torch.Tensor) else q2
        
        # Cart parameters (approximate)
        cart_width = 0.3
        cart_height = 0.2
        pole_length = 1.0  # Adjust based on your lp parameter
        
        # Draw ground
        self.ax.plot([-3, 3], [0, 0], 'k-', linewidth=2)
        
        # Draw cart
        cart = plt.Rectangle((cart_x - cart_width/2, 0), 
                            cart_width, cart_height, 
                            facecolor='blue', edgecolor='black', linewidth=2)
        self.ax.add_patch(cart)
        
        # Draw pole
        pole_x = cart_x + pole_length * np.sin(pole_angle)
        pole_y = cart_height + pole_length * np.cos(pole_angle)
        self.ax.plot([cart_x, pole_x], [cart_height, pole_y], 'r-', linewidth=3)
        
        # Add info text
        self.ax.text(-2.8, 1.8, f'Time: {self.t:.2f}s', fontsize=10)
        self.ax.text(-2.8, 1.6, f'Angle: {np.degrees(pole_angle):.1f}°', fontsize=10)
        
        plt.draw()
        plt.pause(0.001)

def close(self):
    """Close the rendering window"""
    if hasattr(self, 'fig'):
        plt.close(self.fig)
        
        
if __name__ == "__main__":
    try:
        check_env(CartPoleEnv_srk4_torch())
        print("Environment passes all checks!")
    except Exception as e:
        print(f"Environment has issues: {e}")
        import traceback
        traceback.print_exc()




