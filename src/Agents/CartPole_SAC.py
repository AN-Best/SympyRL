import torch
import sys
from pathlib import Path
from stable_baselines3 import SAC
import multiprocessing as mp
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor


parent_folder = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_folder / "Env"))

from Cartpole_SRK4_torch import CartPoleEnv_srk4_torch

def make_env():
    def _init():
        env = CartPoleEnv_srk4_torch()
        env = Monitor(env)
        return env
    return _init

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    
    num_envs = 8
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    
    try:
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=100000,
            batch_size=256,
            device="cuda",
        )
        
        model.learn(total_timesteps=500000)
        model.save("sac_cartpole")
        
        print("Training complete!")
        
    finally:
        env.close()