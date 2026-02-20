# SympyRL Documentation

## Overview
SympyRL is a library designed for reinforcement learning with robust symbolic capabilities. It serves as an efficient symbolic physics engine to simulate various environments, easily integrating with popular machine learning frameworks like PyTorch and JAX.

## Current Capabilities

### 1. Symbolic Physics Engine
The heart of SympyRL lies in its symbolic physics engine that allows dynamic modification of physics simulations, enabling flexibility and deep insights into the underlying mechanics of the systems being modeled.

### 2. Models
SympyRL includes several predefined models that demonstrate its capabilities:
- **CartPole**: A classic reinforcement learning task that showcases stability and control mechanics.
- **Acrobot**: A two-link arm robot model that tests complex dynamics and control strategies.

### 3. Solvers
Our library offers various solvers optimized for performance and accuracy, including:
- Symplectic integrators that maintain geometric properties of the systems.
- Adaptive step-size methods for efficient computation.

### 4. Quick Start Examples
To get started quickly, check out the provided examples that help in implementing reinforcement learning tasks using the CartPole and Acrobot models. You'll see how easy it is to set up an environment, train a model, and visualize the results.

### 5. Roadmap
SympyRL aims to expand its capabilities further in the following areas:
- Enhanced support for GPU-accelerated simulations with batched processing.
- Development of additional pre-trained policies for popular environments.
- Continued integration with leading machine learning frameworks to provide an expansive toolkit for developers.

## Key Features
- **Backend Support**: SympyRL is built to support both PyTorch and JAX, providing users with flexibility in their choice of tools for model training and evaluation.
- **Batched GPU Simulation**: The library supports batched environments, allowing for more efficient learning and simulation processes that leverage modern GPU architectures.
- **Pre-trained Policies**: Users can access pre-trained models that can be fine-tuned or utilized directly for different tasks in the symbolic physics simulations.
