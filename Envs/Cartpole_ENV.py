import jax
import jax.numpy as jnp
from jax import random, jit
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pathlib import Path
import sys

# -------------------------------------------------
# Add Models / Solvers to path
# -------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Models"))
sys.path.insert(0, str(ROOT / "Solvers"))

from CartPole import cartpole_dynamics_batched
from SymplecticRK4 import symplectic_rk4_step

# -------------------------------------------------
# Environment
# -------------------------------------------------
class JAXCartPoleEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dt=0.02):
        super().__init__()

        self.dt = dt
        self.params = (1.0, 0.1, 1.0, 0.01, 9.81)

        # ----- Gym spaces (NUMPY, not JAX) -----
        high = np.array([4.8, np.pi, 10.0, 10.0], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-10.0, high=10.0, shape=(1,), dtype=np.float32
        )

        self.state = np.zeros(4, dtype=np.float32)

    # -------------------------------------------------
    # Reset
    # -------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        self.state = rng.uniform(-0.05, 0.05, size=(4,)).astype(np.float32)
        return self.state, {}

    # -------------------------------------------------
    # Step
    # -------------------------------------------------
    def step(self, action):
        # SB3 gives (1,) action
        u = jnp.asarray(action).reshape(1, 1)
        x = jnp.asarray(self.state).reshape(1, 4)

        # --- JAX physics step ---
        x_next, _ = symplectic_rk4_step(
            f=cartpole_dynamics_batched,
            x=x,
            t=0.0,
            u=u,
            dt=self.dt,
            params=self.params,
        )

        x_next = x_next[0]  # remove batch dim
        self.state = np.array(x_next, dtype=np.float32)

        # ----- Termination -----
        terminated = bool(abs(self.state[0]) > 2.4)
        truncated = False

        reward = 1.0 if not terminated else -100.0

        return self.state, reward, terminated, truncated, {}

