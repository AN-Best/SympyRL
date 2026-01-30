import jax
import jax.numpy as jnp
import numpy as np
import sys
from pathlib import Path
import time

parent_folder = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_folder / "Models"))
sys.path.insert(0, str(parent_folder / "Solvers"))

from CartPole import cartpole_dynamics_batched
from SymplecticRK4 import symplectic_rk4_step

params = jnp.array([1.0, 0.1, 1.0, 0.01, 9.81])
batch_size = 100000
dt = 0.002
T = 10.0
steps = int(T/dt)

# Initial state
rng = np.random.default_rng(42)
x0 = jnp.array(rng.uniform(-0.05, 0.05, (batch_size, 4)))
u = jnp.zeros((batch_size, 1))  # unactuated

# ------------------------
# Fully fused simulation (no energy)
# ------------------------
def run_simulation_memory_efficient(x0, u, dt, params, steps):
    state = x0
    t = 0.0

    def body_fun(i, val):
        state, t = val
        state, t = symplectic_rk4_step(cartpole_dynamics_batched, state, t, u, dt, params)
        return state, t

    state, t = jax.lax.fori_loop(0, steps, body_fun, (state, t))
    return state

# ------------------------
# JIT compile (steps must be static)
# ------------------------
run_simulation_jit = jax.jit(run_simulation_memory_efficient, static_argnums=(4,))

# ------------------------
# First run (compilation)
# ------------------------
print("Compiling and running first simulation...")
start_time = time.time()
x_final = run_simulation_jit(x0, u, dt, params, steps)
jax.device_get(x_final)  # force computation
first_duration = time.time() - start_time
print(f"First run (including JIT compile): {first_duration:.3f} s")

# ------------------------
# Second run (pure execution)
# ------------------------
print("Running second simulation (timed)...")
start_time = time.time()
x_final = run_simulation_jit(x0, u, dt, params, steps)
jax.device_get(x_final)  # force computation
second_duration = time.time() - start_time
print(f"Second run (pure execution): {second_duration:.3f} s")


