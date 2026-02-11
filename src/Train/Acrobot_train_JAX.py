import jax
import jax.numpy as jnp
from typing import Tuple, NamedTuple
from functools import partial

# Import your dynamics and integrator
import sys
from pathlib import Path
parent_folder = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_folder / "Agents"))
sys.path.insert(0, str(parent_folder / "Env"))

from PPO_JAX import make_train
from Acobrot_SRK4_jax import AcrobotEnv

config = {
    "LR": 3e-4,
    "NUM_ENVS": 2048,
    "NUM_STEPS": 10,
    "TOTAL_TIMESTEPS": 5e7,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 32,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.0,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "tanh",
    "ENV_NAME": "AcrobotEnv",
    "ANNEAL_LR": False,
    "NORMALIZE_ENV": True,
    "DEBUG": True,
}
rng = jax.random.PRNGKey(30)
train_jit = jax.jit(make_train(config))
out = train_jit(rng)