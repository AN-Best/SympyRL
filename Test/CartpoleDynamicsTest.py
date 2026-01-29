import jax
import jax.numpy as jnp
import numpy as np
import sys
from pathlib import Path
import functools


# Get the parent folder (one level up)
parent_folder = Path(__file__).resolve().parent.parent

# Add Models and Solvers folders to sys.path
sys.path.insert(0, str(parent_folder / "Models"))
sys.path.insert(0, str(parent_folder / "Solvers"))


# Import your dynamics and RK4 functions
from CartPole import cartpole_dynamics_batched
from SymplecticRK4 import symplectic_rk4_step  

f = lambda x, u: cartpole_dynamics_batched(x, u, params)
solver_jit = jax.jit(symplectic_rk4_step, static_argnums=(0,))

# ------------------------
# Physical parameters
# ------------------------
params = (1.0, 0.1, 1.0, 0.01, 9.81)  # mc, mp, lp, Ip, g

print(jax.devices())

# ------------------------
# Energy function
# ------------------------
def cartpole_energy(x, params):
    """
    Compute total mechanical energy (cart + pole)
    x: [batch,4] -> [q1, q2, u1, u2]
    """
    mc, mp, lp, Ip, g = params
    q1 = x[:,0]
    q2 = x[:,1]
    u1 = x[:,2]
    u2 = x[:,3]

    # Cart kinetic
    T_cart = 0.5 * mc * u1**2

    # Pole kinetic (translational + rotational)
    v_pole_x = u1 + (lp/2) * u2 * jnp.cos(q2)
    v_pole_y = (lp/2) * u2 * jnp.sin(q2)
    T_pole = 0.5 * mp * (v_pole_x**2 + v_pole_y**2) + 0.5 * Ip * u2**2

    # Potential energy (only gravity)
    V_pole = mp * g * (lp/2) * jnp.cos(q2)
    V_cart = 0.0

    E = T_cart + T_pole + V_cart + V_pole
    return E

# ------------------------
# Simulation parameters
# ------------------------
batch_size = 10000
dt = 0.002
steps = int(10.0/dt)

# Random initial states [q1, q2, u1, u2]
rng = np.random.default_rng(42)
x0 = jnp.array(rng.uniform(-0.05,0.05,(batch_size,4)))
t0 = 0.0
u = jnp.zeros((batch_size,1))  # unactuated

# ------------------------
# Record energy
# ------------------------
energies = []

def step(carry, _):
    x, t = carry
    x, t = solver_jit(cartpole_dynamics_batched, x, t, u, dt,params)
    e = cartpole_energy(x, params)
    return (x, t), e

(carry_final), energies = jax.lax.scan(step, (x0, t0), None, length=steps)

energies = jnp.stack(energies)  # [steps, batch]

# ------------------------
# Print energy drift
# ------------------------
E0 = energies[0]
E_final = energies[-1]
drift = jnp.abs(E_final - E0) / E0
for i in range(batch_size):
    print(f"Pole {i}: initial={E0[i]:.5f}, final={E_final[i]:.5f}, relative drift={drift[i]:.5e}")
