import torch
import numpy as np
import sys
from pathlib import Path
import time

parent_folder = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_folder / "src/Models"))
sys.path.insert(0, str(parent_folder / "src/Solvers"))

from Acrobot import acrobot_dynamics_batched_torch
from SymplecticRK4_TORCH import symplectic_rk4_step_torch


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

params = (1.0, 1.0, 1.0/12.0, 9.81)
batch_size = 1000
dt = 0.002
T = 10.0
steps = int(T/dt)

# Initial state
rng = np.random.default_rng(42)
x0 = torch.tensor(rng.uniform(-0.05, 0.05, (batch_size, 4)), dtype=torch.float32, device=device)
u = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)  # unactuated

# ------------------------
# Simulation loop (not compiled - just calls the compiled dynamics)
# ------------------------
def run_simulation_memory_efficient(x0, u, dt, params, steps):
    state = x0
    t = 0.0

    for i in range(steps):
        state, t = symplectic_rk4_step_torch(acrobot_dynamics_batched_torch, state, t, u, dt, params)
    
    return state

# ------------------------
# First run (compiles dynamics on first call)
# ------------------------
print("Running first simulation (compiles dynamics on first call)...")
start_time = time.time()
x_final = run_simulation_memory_efficient(x0, u, dt, params, steps)
torch.cuda.synchronize() if torch.cuda.is_available() else None  # force computation
first_duration = time.time() - start_time
print(f"First run (including dynamic compilation): {first_duration:.3f} s")

# ------------------------
# Second run (pure execution with compiled dynamics)
# ------------------------
print("Running second simulation (timed with compiled dynamics)...")
start_time = time.time()
x_final = run_simulation_memory_efficient(x0, u, dt, params, steps)
torch.cuda.synchronize() if torch.cuda.is_available() else None  # force computation
second_duration = time.time() - start_time
print(f"Second run (pure execution): {second_duration:.3f} s")

print(f"\nFinal state shape: {x_final.shape}")
print(f"Sample final states (first 3):\n{x_final[:3]}")