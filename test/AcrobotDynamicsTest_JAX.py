import jax
import jax.numpy as jnp
import numpy as np
import sys
from pathlib import Path
import time

# Enable float32 for better GPU performance
jax.config.update("jax_enable_x64", False)

parent_folder = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_folder / "src/Models"))
sys.path.insert(0, str(parent_folder / "src/Solvers"))

from Acrobot import acrobot_dynamics_batched_jax
from SymplecticRK4_JAX import symplectic_rk4_step_jax

params = jnp.array([1.0, 1.0, 1.0/12.0, 9.81], dtype=jnp.float32)
batch_size = 1000
dt = 0.002
T = 10.0
steps = int(T/dt)

# Initial state
rng = np.random.default_rng(42)
x0 = jnp.array(rng.uniform(-0.05, 0.05, (batch_size, 4)), dtype=jnp.float32)
u = jnp.zeros((batch_size, 1), dtype=jnp.float32)

# ------------------------
# Fully fused simulation
# ------------------------
def run_simulation_memory_efficient(x0, u, dt, params, steps):
    state = x0
    t = 0.0

    def body_fun(i, val):
        state, t = val
        state, t = symplectic_rk4_step_jax(acrobot_dynamics_batched_jax, state, t, u, dt, params)
        return state, t

    state, t = jax.lax.fori_loop(0, steps, body_fun, (state, t))
    return state

# ------------------------
# JIT compile
# ------------------------
run_simulation_jit = jax.jit(run_simulation_memory_efficient, static_argnums=(4,))

# ------------------------
# First run (compilation)
# ------------------------
print("Compiling and running first simulation...")
start_time = time.time()
x_final = run_simulation_jit(x0, u, dt, params, steps)
x_final.block_until_ready()  # Better than device_get for timing
first_duration = time.time() - start_time
print(f"First run (including JIT compile): {first_duration:.3f} s")

# ------------------------
# Second run (pure execution)
# ------------------------
print("Running second simulation (timed)...")
start_time = time.time()
x_final = run_simulation_jit(x0, u, dt, params, steps)
x_final.block_until_ready()
second_duration = time.time() - start_time
print(f"Second run (pure execution): {second_duration:.3f} s")

print(f"\nFinal state shape: {x_final.shape}")
print(f"Sample final states (first 3):\n{x_final[:3]}")