# SympyRL

**SympyRL** is a research-oriented framework for building *symbolic physics-based reinforcement learning environments* using **SymPy** for analytical dynamics and **PyTorch** for fast numerical simulation and learning.

The goal of the project is to bridge the gap between:

- exact, interpretable dynamics models (via symbolic mechanics), and  
- modern deep reinforcement learning workflows (e.g., Stable-Baselines3, custom PyTorch pipelines).

This makes SympyRL well suited for research in:

- model-based and model-aware RL,
- control of underactuated systems (cartpoles, acrobots, walkers, drones, humanoids),
- differentiable physics and learning-from-dynamics,
- benchmarking RL algorithms on analytically-defined systems.

---

## Key Features

- Symbolic derivation of equations of motion using **SymPy**
- Automatic conversion to efficient numerical functions
- PyTorch-compatible batched dynamics
- GPU-friendly simulation loops
- Gymnasium-style environments
- Designed for continuous control and physical systems
- Modular structure for adding new models and solvers

---

## Repository Structure

```
SympyRL/
├── Models/        # Symbolic system definitions (e.g., CartPole, Acrobot, etc.)
├── Solvers/       # Numerical integrators (e.g., RK4)
├── Environments/ # Gymnasium-style RL environments
├── Agents/       # RL agent wrappers / experiments
├── scripts/      # Training and evaluation scripts
└── tests/        # (planned) automated tests
```

---

## Example: CartPole Swing-Up (PyTorch)

```python
from Models.CartPole import CartPoleDynamicsBatched
from Solvers.RK4 import RK4Integrator

model = CartPoleDynamicsBatched(device="cuda")
solver = RK4Integrator(model, dt=0.02)

x_next = solver.step(x, u)
```

Training with Stable-Baselines3:

```python
from stable_baselines3 import SAC
from Environments.CartPoleEnv import CartPoleEnv

env = CartPoleEnv(device="cuda")
model = SAC("MlpPolicy", env)
model.learn(1_000_000)
```

---

## Installation

```bash
git clone https://github.com/AN-Best/SympyRL.git
cd SympyRL
pip install -r requirements.txt
```

Dependencies include:

- sympy
- numpy
- torch
- gymnasium
- stable-baselines3

---

## Design Philosophy

SympyRL prioritizes:

- **Correctness** – dynamics derived symbolically
- **Transparency** – inspectable equations of motion
- **Performance** – vectorized, GPU-compatible execution
- **Research flexibility** – easy modification of models and controllers

Unlike traditional simulators (e.g., MuJoCo, Bullet), SympyRL allows full access to the analytical structure of the system, enabling:

- custom linearizations,
- exact Jacobians,
- hybrid model-based + model-free RL,
- differentiable control pipelines.

---

## Current Status

This project is under active development. Implemented systems include:

- CartPole (swing-up)
- Acrobot (in progress)

Planned additions:

- walkers
- quadrotors
- humanoid models
- JAX-native backend
- differentiable integrators
- benchmarking suite

---

## Roadmap

- [ ] Add automated test suite
- [ ] Add documentation website
- [ ] Add more physical models
- [ ] JAX backend
- [ ] Model-based RL examples
- [ ] Reproducible benchmarks

---

## Citation

If you use this software in research, please cite:

```
@software{sympyrl,
  title  = {SympyRL: Symbolic Physics for Reinforcement Learning},
  author = {Best, Aaron},
  year   = {2026},
  url    = {https://github.com/AN-Best/SympyRL}
}
```

---

## License
TO DO

---

## Contact

Aaron Best  
GitHub: https://github.com/AN-Best

---

## Acknowledgements

Inspired by:

- SymPy Mechanics
- OpenAI Gym / Gymnasium
- Stable-Baselines3
- opty
- Model-based control and optimal control literature

