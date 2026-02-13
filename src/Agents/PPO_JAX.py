"""
PPO Training for Acrobot JAX Environment
Adapted from PureJaxRL for continuous action spaces

This is a single-file implementation following PureJaxRL's philosophy.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple
from flax.training.train_state import TrainState
import distrax
from functools import partial
import sys
from pathlib import Path
parent_folder = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_folder / "Env"))

# Import your environment
from Acrobot_SRK4_jax import AcrobotEnv, EnvParams, EnvState



# ============================================================================
# Network Architecture
# ============================================================================

class ActorCriticContinuous(nn.Module):
    """Actor-Critic network for continuous action spaces."""
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        
        # Shared feature extractor
        actor = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor = activation(actor)
        actor = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor)
        actor = activation(actor)
        
        # Actor head: mean and log_std
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor)
        # Learnable log_std (not state-dependent)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        
        # Critic network
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return actor_mean, actor_logtstd, jnp.squeeze(critic, axis=-1)


# ============================================================================
# Transition Storage
# ============================================================================

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


# ============================================================================
# Main Training Function
# ============================================================================

def make_train(config):
    """Create the training function."""
    
    # Environment setup
    env = AcrobotEnv()
    env_params = EnvParams()
    
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    
    def linear_schedule(count):
        """Learning rate schedule."""
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        """The main training loop - JIT compiled for speed."""
        
        # INIT NETWORK
        network = ActorCriticContinuous(
            action_dim=env.action_dim,
            activation=config["ACTIVATION"]
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.obs_dim)
        network_params = network.init(_rng, init_x)
        
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        env_state, obsv = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                actor_mean, actor_logtstd, value = network.apply(
                    train_state.params, last_obs
                )
                # Sample action from diagonal Gaussian
                pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                
                # Clip action to [-1, 1] (environment bounds)
                action = jnp.clip(action, -1.0, 1.0)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                next_env_state, obsv, reward, done, info = jax.vmap(
                    env.step_with_autoreset, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
                env_state = next_env_state
                
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            _, _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        actor_mean, actor_logtstd, value = network.apply(
                            params, traj_batch.obs
                        )
                        pi = distrax.MultivariateNormalDiag(
                            actor_mean, jnp.exp(actor_logtstd)
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


# ============================================================================
# Training Configuration & Execution
# ============================================================================

if __name__ == "__main__":
    config = {
        "LR": 3e-4,
        "NUM_ENVS": 2048,  
        "NUM_STEPS": 256,
        "TOTAL_TIMESTEPS": 40e8,  
        "UPDATE_EPOCHS": 10,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,  
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": True,
    }

    print("=" * 70)
    print("PureJaxRL-style PPO Training for Acrobot")
    print("=" * 70)
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key:20s}: {value}")
    
    print(f"\nDerived settings:")
    num_updates = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    steps_per_update = config["NUM_ENVS"] * config["NUM_STEPS"]
    print(f"  {'NUM_UPDATES':20s}: {num_updates}")
    print(f"  {'Steps per update':20s}: {steps_per_update:,}")
    print(f"  {'Total steps':20s}: {num_updates * steps_per_update:,}")
    
    print("\n" + "-" * 70)
    print("Starting training...")
    print("-" * 70 + "\n")

    rng = jax.random.PRNGKey(42)
    train_jit = jax.jit(make_train(config))
    
    # Warmup compilation
    print("Compiling... (this may take a minute)")
    import time
    start_compile = time.time()
    out = train_jit(rng)
    compile_time = time.time() - start_compile
    print(f"Compilation completed in {compile_time:.1f}s\n")

    # Run training (already compiled, so this is fast!)
    print("Training...")
    start_train = time.time()
    rng = jax.random.PRNGKey(30)
    out = train_jit(rng)
    train_time = time.time() - start_train
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Wall clock time: {train_time:.1f}s")
    print(f"Total steps: {num_updates * steps_per_update:,}")
    print(f"Steps per second: {(num_updates * steps_per_update) / train_time:,.0f}")
    
    # Print some final metrics
    metrics = out["metrics"]
    print(f"\nMetrics shape: {jax.tree.map(lambda x: x.shape, metrics)}")
    print("\nTraining finished! Use the returned 'out' dict to access:")
    print("  - out['runner_state']: Final training state (network params, env state)")
    print("  - out['metrics']: Training metrics over time")


# ========================================================================
    # Visualize Trained Policy
    # ========================================================================
    print("\n" + "=" * 70)
    print("Visualizing Trained Policy")
    print("=" * 70)
    
    # Extract trained policy
    train_state = out["runner_state"][0]
    network = ActorCriticContinuous(action_dim=1, activation=config["ACTIVATION"])
    
    # Create single environment for visualization
    env = AcrobotEnv()
    env_params = EnvParams()
    
    # Run a single episode
    rng = jax.random.PRNGKey(100)
    rng, reset_rng = jax.random.split(rng)
    state, obs = env.reset(reset_rng, env_params)
    
    print("\nRunning episode...")
    episode_rewards = []
    episode_states = []
    episode_actions = []
    
    max_steps = 2500  # 5 seconds at 0.002 dt
    
    for step in range(max_steps):
        # Get action from policy
        actor_mean, actor_logtstd, value = network.apply(train_state.params, obs)
        # Use mean action (deterministic policy for visualization)
        action = actor_mean
        action = jnp.clip(action, -1.0, 1.0)
        
        # Step environment
        rng, step_rng = jax.random.split(rng)
        state, obs, reward, done, info = env.step(step_rng, state, action, env_params)
        
        episode_rewards.append(float(reward))
        episode_states.append(state)
        episode_actions.append(float(action[0]))
        
        if done:
            print(f"Episode finished at step {step+1}")
            break
    
    episode_rewards = jnp.array(episode_rewards)
    episode_actions = jnp.array(episode_actions)
    
    print(f"\nEpisode Statistics:")
    print(f"  Length: {len(episode_rewards)} steps")
    print(f"  Total reward: {episode_rewards.sum():.2f}")
    print(f"  Mean reward: {episode_rewards.mean():.2f}")
    print(f"  Max reward: {episode_rewards.max():.2f}")
    print(f"  Min reward: {episode_rewards.min():.2f}")
    
    # Visualize with matplotlib
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        
        # Plot rewards
        axes[0].plot(episode_rewards)
        axes[0].set_ylabel('Reward')
        axes[0].set_title('Episode Rewards')
        axes[0].grid(True)
        axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Plot actions
        axes[1].plot(episode_actions)
        axes[1].set_ylabel('Action (Torque)')
        axes[1].set_title('Policy Actions')
        axes[1].grid(True)
        axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1].set_ylim(-1.1, 1.1)
        
        # Plot joint angles
        q1_vals = [float(s.q1) for s in episode_states]
        q2_vals = [float(s.q2) for s in episode_states]
        axes[2].plot(q1_vals, label='q1 (joint 1)')
        axes[2].plot(q2_vals, label='q2 (joint 2)')
        axes[2].set_ylabel('Angle (rad)')
        axes[2].set_xlabel('Step')
        axes[2].set_title('Joint Angles')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig('acrobot_training_results.png', dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to 'acrobot_training_results.png'")
        plt.show()
        
    except ImportError:
        print("\nMatplotlib not available, skipping visualization plots")
    
    # Animate the acrobot motion
    print("\nCreating animation...")
    try:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title('Trained Acrobot Policy')
        
        line, = ax.plot([], [], 'o-', lw=2, markersize=8, color='blue')
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        reward_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
        
        def init():
            line.set_data([], [])
            time_text.set_text('')
            reward_text.set_text('')
            return line, time_text, reward_text
        
        def animate(i):
            if i >= len(episode_states):
                return line, time_text, reward_text
            
            state = episode_states[i]
            q1, q2 = float(state.q1), float(state.q2)
            
            # Forward kinematics
            L1, L2 = 1.0, 1.0
            x0, y0 = 0.0, 0.0
            x1 = L1 * jnp.sin(q1)
            y1 = -L1 * jnp.cos(q1)
            x2 = x1 + L2 * jnp.sin(q1 + q2)
            y2 = y1 - L2 * jnp.cos(q1 + q2)
            
            line.set_data([x0, x1, x2], [y0, y1, y2])
            time_text.set_text(f'Step: {i}')
            reward_text.set_text(f'Reward: {episode_rewards[i]:.2f}')
            return line, time_text, reward_text
        
        # Create animation (show every 5th frame for speed)
        skip = 5
        anim = animation.FuncAnimation(
            fig, animate, init_func=init,
            frames=range(0, len(episode_states), skip),
            interval=20, blit=True
        )
        
        # Save animation
        anim.save('acrobot_trained_policy.gif', writer='pillow', fps=30)
        print("Animation saved to 'acrobot_trained_policy.gif'")
        plt.close()
        
    except Exception as e:
        print(f"\nCould not create animation: {e}")
    
    print("\n" + "=" * 70)
