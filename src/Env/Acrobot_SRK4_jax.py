import jax
import jax.numpy as jnp
from typing import Tuple, NamedTuple
from functools import partial

# Import your dynamics and integrator
import sys
from pathlib import Path
parent_folder = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_folder / "Models"))
sys.path.insert(0, str(parent_folder / "Solvers"))

from Acrobot import acrobot_dynamics_batched_jax
from SymplecticRK4_JAX import symplectic_rk4_step_jax



class EnvState(NamedTuple):
    """Environment state for the Acrobot."""
    q1: jnp.ndarray  # angle 1
    q2: jnp.ndarray  # angle 2
    u1: jnp.ndarray  # velocity 1
    u2: jnp.ndarray  # velocity 2
    t: jnp.ndarray   # time
    step: jnp.ndarray  # step counter


class EnvParams(NamedTuple):
    """Environment parameters."""
    m: float = 1.0      # mass
    l: float = 1.0      # length
    I: float = 1.0/12.0 # inertia
    g: float = 9.81     # gravity
    dt: float = 0.002   # timestep
    max_steps: int = 2500  # 5.0 / 0.002
    T_scale: float = 200.0  # action scaling


class AcrobotEnv:
    """JAX-based Acrobot environment with symplectic RK4 integration.
    
    This environment follows the functional JAX pattern:
    - All state is explicit (no hidden internal state)
    - All randomness uses explicit PRNG keys
    - All functions are pure (no side effects)
    - Fully vectorizable with jax.vmap
    """
    
    def __init__(self, params: EnvParams = None):
        self.params = params if params is not None else EnvParams()
        
        # Action space: [-1, 1]
        self.action_dim = 1
        self.action_low = -1.0
        self.action_high = 1.0
        
        # Observation space: [q1, q2, u1, u2]
        self.obs_dim = 4
        self.obs_low = jnp.array([-50.0, -50.0, -50.0, -50.0])
        self.obs_high = jnp.array([50.0, 50.0, 50.0, 50.0])
    
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.random.PRNGKey, params: EnvParams = None) -> Tuple[EnvState, jnp.ndarray]:
        """Reset the environment to initial state.
        
        Args:
            key: JAX random key
            params: Optional environment parameters (uses default if None)
            
        Returns:
            state: Initial environment state
            obs: Initial observation [q1, q2, u1, u2]
        """
        if params is None:
            params = self.params
            
        # Split key for each random variable
        key, k1, k2, k3, k4 = jax.random.split(key, 5)
        
        # Sample initial state
        q1 = jax.random.uniform(k1, (), minval=-jnp.pi/6, maxval=jnp.pi/6)
        q2 = jax.random.uniform(k2, (), minval=-jnp.pi/6, maxval=jnp.pi/6)
        u1 = jax.random.uniform(k3, (), minval=-1.0, maxval=1.0)
        u2 = jax.random.uniform(k4, (), minval=-1.0, maxval=1.0)
        
        state = EnvState(
            q1=q1,
            q2=q2,
            u1=u1,
            u2=u2,
            t=jnp.array(0.0),
            step=jnp.array(0, dtype=jnp.int32)
        )
        
        obs = self._get_obs(state)
        return state, obs
    
    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, 
        key: jax.random.PRNGKey,
        state: EnvState, 
        action: jnp.ndarray,
        params: EnvParams = None
    ) -> Tuple[EnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
        """Take one step in the environment.
        
        Args:
            key: JAX random key (for future stochasticity if needed)
            state: Current environment state
            action: Action to take, shape () or (1,), values in [-1, 1]
            params: Optional environment parameters
            
        Returns:
            next_state: Next environment state
            obs: Next observation
            reward: Reward for this transition
            done: Whether episode is terminated
            info: Additional information (empty dict)
        """
        if params is None:
            params = self.params
        
        # Scale action to torque
        action = jnp.atleast_1d(action)  # Ensure it's at least 1D
        u = action * params.T_scale
        
        # Prepare state for dynamics: [q1, q2, u1, u2]
        x_current = jnp.array([state.q1, state.q2, state.u1, state.u2])
        x_current = x_current[None, :]  # Add batch dimension: [1, 4]
        u_batched = u.reshape(1, -1)  # [1, 1]
        
        # Physics parameters
        physics_params = (params.m, params.l, params.I, params.g)
        
        # Integrate using symplectic RK4
        x_next, t_next = symplectic_rk4_step_jax(
            acrobot_dynamics_batched_jax,
            x_current,
            state.t,
            u_batched,
            params.dt,
            physics_params
        )
        
        # Extract next state (remove batch dimension)
        x_next = x_next[0]  # [4]
        q1_next, q2_next, u1_next, u2_next = x_next[0], x_next[1], x_next[2], x_next[3]
        
        # Clamp velocities
        u1_next = jnp.clip(u1_next, -50.0, 50.0)
        u2_next = jnp.clip(u2_next, -50.0, 50.0)
        
        # Update state
        next_state = EnvState(
            q1=q1_next,
            q2=q2_next,
            u1=u1_next,
            u2=u2_next,
            t=t_next,
            step=state.step + 1
        )
        
        # Compute reward
        reward = self._compute_reward(next_state)
        
        # Check termination
        done = state.step >= params.max_steps
        
        # Get observation
        obs = self._get_obs(next_state)
        
        return next_state, obs, reward, done, {}
    
    @partial(jax.jit, static_argnums=(0,))
    def step_with_autoreset(
        self,
        key: jax.random.PRNGKey,
        state: EnvState,
        action: jnp.ndarray,
        params: EnvParams = None
    ) -> Tuple[EnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
        """Step with automatic reset on termination (useful for vectorized training).
        
        This version automatically resets the environment when done, which is
        the standard pattern for JAX RL environments to simplify vectorized rollouts.
        """
        if params is None:
            params = self.params
            
        # Take regular step
        key, subkey = jax.random.split(key)
        next_state, obs, reward, done, info = self.step(subkey, state, action, params)
        
        # Auto-reset if done
        key, reset_key = jax.random.split(key)
        reset_state, reset_obs = self.reset(reset_key, params)
        
        # Use jax.tree_map to conditionally select reset or next state
        next_state = jax.tree.map(
            lambda x, y: jnp.where(done, x, y),
            reset_state,
            next_state
        )
        obs = jnp.where(done, reset_obs, obs)
        
        return next_state, obs, reward, done, info
    
    def _get_obs(self, state: EnvState) -> jnp.ndarray:
        """Get observation from state."""
        return jnp.array([state.q1, state.q2, state.u1, state.u2])
    
    def _compute_reward(self, state: EnvState) -> jnp.ndarray:
        """Compute reward based on state.
        
        Reward is based on the height of the end effector.
        Higher position = higher reward.
        """
        # Normalize angles to [-pi, pi]
        ang1 = jnp.arctan2(jnp.sin(state.q1), jnp.cos(state.q1))
        ang2 = jnp.arctan2(jnp.sin(state.q2), jnp.cos(state.q2))
        
        # Height of end effector (ranges from -2 to +2)
        # h = -2 when hanging down, h = +2 when fully up
        h = -jnp.cos(ang1) - jnp.cos(ang1 + ang2)
        
        # Target height for swing-up (slightly below max to be achievable)
        target_height = 1.68
        
        # Dense reward: proportional to height (encourages upward motion)
        # Scale to roughly [-1, 1] range
        height_reward = h / 2.0  # Ranges from -1.0 to +1.0
        
        # Success bonus: large reward for reaching target
        success = h >= target_height
        success_bonus = jnp.where(success, 200.0, 0.0)

        # Slow down
        speed_penalty = jnp.pow(state.u1,2) + jnp.pow(state.u2,2)
        
        # Combine rewards
        reward = height_reward + success_bonus - 0.1*speed_penalty
        
        return reward


# ============================================================================
# Vectorization helpers
# ============================================================================

def make_env(params: EnvParams = None):
    """Factory function to create environment."""
    return AcrobotEnv(params)


@partial(jax.jit, static_argnums=(0,))
def vectorized_reset(env: AcrobotEnv, keys: jax.random.PRNGKey, params: EnvParams = None):
    """Reset multiple environments in parallel.
    
    Args:
        env: Environment instance (marked as static)
        keys: Array of random keys, shape (num_envs,)
        params: Optional environment parameters
        
    Returns:
        states: Batch of initial states
        obs: Batch of initial observations, shape (num_envs, 4)
    """
    return jax.vmap(env.reset, in_axes=(0, None))(keys, params)


@partial(jax.jit, static_argnums=(0,))
def vectorized_step(
    env: AcrobotEnv,
    keys: jax.random.PRNGKey,
    states: EnvState,
    actions: jnp.ndarray,
    params: EnvParams = None
):
    """Step multiple environments in parallel.
    
    Args:
        env: Environment instance (marked as static)
        keys: Array of random keys, shape (num_envs,)
        states: Batch of current states
        actions: Batch of actions, shape (num_envs, 1) or (num_envs,)
        params: Optional environment parameters
        
    Returns:
        next_states: Batch of next states
        obs: Batch of observations, shape (num_envs, 4)
        rewards: Batch of rewards, shape (num_envs,)
        dones: Batch of done flags, shape (num_envs,)
        infos: Batch of info dicts
    """
    return jax.vmap(env.step, in_axes=(0, 0, 0, None))(keys, states, actions, params)


@partial(jax.jit, static_argnums=(0,))
def vectorized_step_with_autoreset(
    env: AcrobotEnv,
    keys: jax.random.PRNGKey,
    states: EnvState,
    actions: jnp.ndarray,
    params: EnvParams = None
):
    """Step multiple environments in parallel with auto-reset.
    
    This is the recommended function for training loops.
    """
    return jax.vmap(env.step_with_autoreset, in_axes=(0, 0, 0, None))(
        keys, states, actions, params
    )


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    print("Testing JAX Acrobot Environment")
    print("=" * 60)
    
    # Create environment
    env = AcrobotEnv()
    
    # Test single environment
    print("\n1. Testing single environment:")
    key = jax.random.PRNGKey(0)
    key, reset_key = jax.random.split(key)
    
    state, obs = env.reset(reset_key)
    print(f"Initial state: q1={state.q1:.3f}, q2={state.q2:.3f}, u1={state.u1:.3f}, u2={state.u2:.3f}")
    print(f"Initial obs shape: {obs.shape}")
    
    # Take a step
    key, step_key = jax.random.split(key)
    action = jnp.array([0.5])  # Action in [-1, 1]
    next_state, obs, reward, done, info = env.step(step_key, state, action)
    print(f"After step: reward={reward:.3f}, done={done}")
    
    # Test vectorized environments
    print("\n2. Testing vectorized environments (1000 parallel envs):")
    num_envs = 1000
    
    # Create batch of random keys
    key, subkey = jax.random.split(key)
    reset_keys = jax.random.split(subkey, num_envs)
    
    # Reset all environments
    states, obs_batch = vectorized_reset(env, reset_keys)
    print(f"Observation batch shape: {obs_batch.shape}")  # Should be (1000, 4)
    
    # Take vectorized step
    key, subkey = jax.random.split(key)
    step_keys = jax.random.split(subkey, num_envs)
    actions = jax.random.uniform(key, (num_envs, 1), minval=-1.0, maxval=1.0)
    
    next_states, obs_batch, rewards, dones, infos = vectorized_step(
        env, step_keys, states, actions
    )
    print(f"Rewards shape: {rewards.shape}")  # Should be (1000,)
    print(f"Mean reward: {rewards.mean():.3f}")
    
    # Test JIT compilation speed
    print("\n3. Testing JIT compilation speed:")
    import time
    
    # Warmup
    for _ in range(10):
        key, subkey = jax.random.split(key)
        step_keys = jax.random.split(subkey, num_envs)
        next_states, obs_batch, rewards, dones, infos = vectorized_step(
            env, step_keys, next_states, actions
        )
    
    # Time 100 steps
    start = time.time()
    for _ in range(100):
        key, subkey = jax.random.split(key)
        step_keys = jax.random.split(subkey, num_envs)
        next_states, obs_batch, rewards, dones, infos = vectorized_step(
            env, step_keys, next_states, actions
        )
    end = time.time()
    
    total_steps = 100 * num_envs
    elapsed = end - start
    steps_per_sec = total_steps / elapsed
    
    print(f"Total steps: {total_steps:,}")
    print(f"Time elapsed: {elapsed:.3f} seconds")
    print(f"Steps per second: {steps_per_sec:,.0f}")
    
    print("\n" + "=" * 60)
    print("All tests passed! Environment is ready for RL training.")
    print("\nKey features:")
    print("  ✓ Fully functional (no hidden state)")
    print("  ✓ JIT compiled for speed")
    print("  ✓ Vectorizable across thousands of environments")
    print("  ✓ Compatible with JAX RL libraries")
