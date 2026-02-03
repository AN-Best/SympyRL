# SympyRL

A symbolic physics engine for reinforcement learning with automatic code generation for PyTorch and JAX backends.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## Overview

SympyRL bridges the gap between exact, interpretable dynamics models and modern deep reinforcement learning. Define your system symbolically with SymPy's mechanics module, and SympyRL automatically generates efficient, batched dynamics functions for both **PyTorch** and **JAX**.

```python
# Symbolic dynamics → batched GPU simulation in one line
from Models.CartPole import cartpole_dynamics_batched_jax
from Solvers.SymplecticRK4_JAX import symplectic_rk4_step_jax

x_next, t_next = symplectic_rk4_step_jax(
    cartpole_dynamics_batched_jax, x, t, u, dt, params
)
```

---

## Key Features

- **Symbolic derivation** — Equations of motion via SymPy's Lagrangian/Kane's method
- **Dual backend** — Same model compiles to PyTorch or JAX
- **Batched simulation** — `torch.vmap` and `jax.vmap` for parallel environments
- **Symplectic integrators** — Energy-preserving 4th-order Gauss-Legendre
- **GPU acceleration** — CUDA (PyTorch) and XLA (JAX) support
- **JIT compilation** — `torch.compile` and `jax.jit` for maximum speed

---

## Current Implementation

### Models

| Model | DOF | Backend | Status |
|-------|-----|---------|--------|
| CartPole | 2 | PyTorch + JAX | ✅ Complete |
| Acrobot | 2 | PyTorch | ✅ Complete |

### Solvers

| Solver | Order | Type | Backend | Status |
|--------|-------|------|---------|--------|
| Symplectic RK4 (Gauss-Legendre) | 4 | Implicit | PyTorch | ✅ Complete |
| Symplectic RK4 (Gauss-Legendre) | 4 | Implicit | JAX | ✅ Complete |
| Implicit Midpoint | 2 | Implicit | JAX | ✅ Complete |
| RK4 | 4 | Explicit | NumPy | ✅ Complete |

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              SymPy Symbolic Mechanics                        │
│     (ReferenceFrame, RigidBody, System, KanesMethod)        │
└─────────────────────────┬───────────────────────────────────┘
                          │ lambdify
            ┌─────────────┴─────────────┐
            ▼                           ▼
┌───────────────────────┐   ┌───────────────────────┐
│   PyTorch Backend     │   │     JAX Backend       │
│  • Custom TorchPrinter│   │  • modules="jax"      │
│  • torch.vmap         │   │  • jax.vmap           │
│  • torch.compile      │   │  • jax.jit            │
└───────────────────────┘   └───────────────────────┘
            │                           │
            ▼                           ▼
┌───────────────────────┐   ┌───────────────────────┐
│  Symplectic RK4       │   │  Symplectic RK4       │
│  (Gauss-Legendre)     │   │  + jax.lax.fori_loop  │
└───────────────────────┘   └───────────────────────┘
```

---

## Quick Start

### CartPole with JAX (1000 parallel environments)

```python
import jax
import jax.numpy as jnp
from Models.CartPole import cartpole_dynamics_batched_jax
from Solvers.SymplecticRK4_JAX import symplectic_rk4_step_jax

# Parameters: [cart_mass, pole_mass, pole_length, pole_inertia, gravity]
params = jnp.array([1.0, 0.1, 1.0, 0.01, 9.81])

# Initial states: [batch, 4] = [x, theta, x_dot, theta_dot]
x0 = jnp.zeros((1000, 4))
u = jnp.zeros((1000, 1))  # No control input

# JIT-compiled simulation loop
@jax.jit
def simulate(x0, u, dt, params, steps):
    def body(i, val):
        state, t = val
        return symplectic_rk4_step_jax(
            cartpole_dynamics_batched_jax, state, t, u, dt, params
        )
    return jax.lax.fori_loop(0, steps, body, (x0, 0.0))

x_final, _ = simulate(x0, u, dt=0.002, params=params, steps=5000)
```

### CartPole with PyTorch

```python
import torch
from Models.CartPole import cartpole_dynamics_batched_torch
from Solvers.SymplecticRK4_TORCH import symplectic_rk4_step_torch

params = (1.0, 0.1, 1.0, 0.01, 9.81)
x = torch.randn(1000, 4, device='cuda')
u = torch.zeros(1000, 1, device='cuda')

x_next, t_next = symplectic_rk4_step_torch(
    cartpole_dynamics_batched_torch, x, 0.0, u, 0.002, params
)
```

### Acrobot with PyTorch

```python
import torch
from Models.Acrobot import AcrobotDynamicsBatched

# Parameters: [mass, length, inertia, gravity]
params = [1.0, 1.0, 0.083, 9.81]
dynamics = AcrobotDynamicsBatched(params).cuda()

x = torch.randn(1000, 4, device='cuda')  # [q1, q2, u1, u2]
u = torch.zeros(1000, 1, device='cuda')  # Torque at elbow

dx = dynamics(x, u)  # Returns [u1, u2, u1_dot, u2_dot]
```

---

## Project Structure

```
SympyRL/
├── src/
│   ├── Models/
│   │   ├── CartPole.py      # SymPy model + PyTorch/JAX lambdify
│   │   └── Acrobot.py       # SymPy model + PyTorch lambdify
│   ├── Solvers/
│   │   ├── SymplecticRK4_JAX.py    # 4th-order Gauss-Legendre (JAX)
│   │   ├── SymplecticRK4_TORCH.py  # 4th-order Gauss-Legendre (PyTorch)
│   │   ├── ImplicitMidpoint_JAX.py # 2nd-order implicit midpoint
│   │   └── RK4.py                  # Classic explicit RK4
│   └── Env/                 # (planned) Gymnasium environments
├── test/
│   ├── CartpoleDynamicsTest_JAX.py
│   └── CartpoleDynamicsTest_TORCH.py
└── readme.md
```

---

## Design Philosophy

SympyRL prioritizes:

1. **Correctness** — Dynamics derived symbolically from first principles
2. **Transparency** — Inspectable equations of motion (not black-box XML)
3. **Performance** — Batched, JIT-compiled, GPU-accelerated
4. **Research flexibility** — Easy to add custom models and integrators

Unlike MuJoCo or PyBullet, SympyRL gives you full access to the analytical structure:
- Exact mass matrices and Coriolis terms
- Symbolic Jacobians for linearization
- Custom constraint handling
- Differentiable dynamics for model-based RL

---

## Roadmap

### Completed ✅

- [x] SymPy → PyTorch lambdify pipeline with custom printer
- [x] SymPy → JAX lambdify pipeline
- [x] Batched dynamics (`torch.vmap`, `jax.vmap`)
- [x] Symplectic RK4 integrator (both backends)
- [x] CartPole model (PyTorch + JAX)
- [x] Acrobot model (PyTorch)
- [x] JIT compilation (`torch.compile`, `jax.jit`)

### In Progress 🔄

- [ ] Gymnasium environment wrappers
- [ ] Stable-Baselines3 integration
- [ ] Acrobot JAX backend

### Future Work 📋

#### Phase 1: Core Infrastructure
- [ ] Base `Model` class for standardized interface
- [ ] Base `Environment` class (Gymnasium-compatible)
- [ ] VecEnv support for parallel training
- [ ] Automated testing suite

#### Phase 2: More Models
- [ ] Pendulum (1 DOF, validation)
- [ ] Double Pendulum
- [ ] Single Leg (3 DOF, 6 Hill-type muscles)
- [ ] Single Arm (4 DOF, 8 muscles)
- [ ] Quadrotor (6 DOF)

#### Phase 3: Musculoskeletal Extension
- [ ] Hill-type muscle model (activation dynamics, force-length, force-velocity)
- [ ] Compliant contact model
- [ ] Bipedal walker (12+ DOF, 12+ muscles)
- [ ] Sarcopenia parameterization (aging/muscle weakness simulation)
- [ ] Exoskeleton assistance model

#### Phase 4: RL Integration
- [ ] PPO/SAC training examples
- [ ] Model-based RL examples (MBPO, Dreamer-style)
- [ ] Reward shaping utilities
- [ ] Domain randomization support
- [ ] Teacher-student distillation

#### Phase 5: Advanced Features
- [ ] End-to-end differentiable simulation
- [ ] Implicit differentiation through dynamics
- [ ] Sim-to-real transfer utilities
- [ ] Visualization and rendering

#### Phase 6: Release
- [ ] pip-installable package
- [ ] Documentation website
- [ ] Tutorial notebooks
- [ ] Benchmark suite
- [ ] Paper

---

## Installation

```bash
git clone https://github.com/AN-Best/SympyRL.git
cd SympyRL
pip install -r requirements.txt
```

### Dependencies

**Core:**
- sympy
- numpy

**PyTorch backend:**
- torch (with CUDA for GPU)

**JAX backend:**
- jax
- jaxlib (with CUDA for GPU)

**RL (planned):**
- gymnasium
- stable-baselines3

---

## Citation

```bibtex
@software{sympyrl,
  title  = {SympyRL: Symbolic Physics for Reinforcement Learning},
  author = {Best, Aaron},
  year   = {2026},
  url    = {https://github.com/AN-Best/SympyRL}
}
```

---

## License

MIT License (see LICENSE file)

---

## Contact

Aaron Best  
GitHub: [@AN-Best](https://github.com/AN-Best)

---

## Acknowledgements

- [SymPy](https://www.sympy.org/) — Symbolic mathematics
- [PyTorch](https://pytorch.org/) — Deep learning framework
- [JAX](https://github.com/google/jax) — Composable transformations
- [Gymnasium](https://gymnasium.farama.org/) — RL environment interface
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — RL algorithms
