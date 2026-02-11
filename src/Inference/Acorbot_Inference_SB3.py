from stable_baselines3 import PPO
import sys
from pathlib import Path
parent_folder = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_folder / "Env"))

from Acrobot_SRK4_torch import AcrobotEnv_srk4_torch
import time

# Load trained model
env = AcrobotEnv_srk4_torch()
model = PPO.load("ppo_acrobot")

# Run and render
obs, info = env.reset()
for _ in range(2500):
    env.render()
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.001)  # Slow down for visualization
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()