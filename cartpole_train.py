# cartpole_train_sbx.py
import os
import sys
from pathlib import Path

import gymnasium as gym
from sbx import SAC
import numpy as np

# -----------------------------------
# Add your environment folder to path
# -----------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent / "Envs"))

from Cartpole_ENV import JAXCartPoleEnv  # your fixed environment

# -----------------------------------
# Register Gym environment
# -----------------------------------
gym.envs.registration.register(
    id="JAXCartPole-v0",
    entry_point="Cartpole_ENV:JAXCartPoleEnv",
    max_episode_steps=500,
)

# -----------------------------------
# Training parameters
# -----------------------------------
total_timesteps = 100_000
tensorboard_log = "./logs/jax_cartpole"
os.makedirs(tensorboard_log, exist_ok=True)
device = "cuda"

# -----------------------------------
# Create environment
# -----------------------------------
env = gym.make("JAXCartPole-v0")

# -----------------------------------
# Instantiate SAC agent
# -----------------------------------
model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=tensorboard_log,
    device=device,
)

# -----------------------------------
# Train
# -----------------------------------
model.learn(total_timesteps=total_timesteps)
print("Training complete!")

# -----------------------------------
# Save model
# -----------------------------------
model_path = os.path.join(tensorboard_log, "jax_cartpole_sac.zip")
model.save(model_path)
print(f"Model saved at: {model_path}")
