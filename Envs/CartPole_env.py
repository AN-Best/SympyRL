import jax
import jax.numpy as jnp
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

parent_folder = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_folder / "Models"))
sys.path.insert(0, str(parent_folder / "Solvers"))

from CartPole import cartpole_dynamics_batched
from SymplecticRK4 import symplectic_rk4_step

class CartPoleEnv:
    def __init__(self,dt = 0.002, max_steps = 2500, batch_size = 100):
        
        #Physical parameters
        mc = 5.0
        mp = 2.0
        lp = 1.0
        Ip = (1.0/12.0)*mp*(lp**2)
        g = 9.81

        self.params = (mc, mp, lp, Ip, g)

        #Time step and episode length
        self.dt = dt
        self.max_steps = max_steps
        self.batch_size = batch_size

        #State Bounds
        self.state_bounds = jnp.array([
            [-3.0, 3.0],
            [-jnp.inf, jnp.inf],
            [-10.0, 10.0],
            [-10.0, 10.0]
        ])

        #Action bounds
        self.action_bounds = jnp.array([[-1, 1]])

        #Dimensions
        self.state_dim = 4
        self.action_dim = 1

        #randomization
        self.rng_key = jax.random.PRNGKey(0)

    def reset(self):

        self.rng_key, subkey = jax.random.split(self.rng_key)

        #Initial state
        x0 = jnp.zeros((self.batch_size,self.state_dim))
        x0 = x0.at[:, 1].set(jnp.pi) #Hanging down

        #Add a little noise
        noise = 0.01*jax.random.normal(subkey,shape = (self.batch_size,self.state_dim))

        self.state = x0 + noise
        self.t = 0.0
        self.step_count = 0

        return self.state
    
    def step(self,actions,max_force = 100.0):
        
        u = actions*max_force

        #Integrate
        self.state, self.t = symplectic_rk4_step(cartpole_dynamics_batched,
                                       self.state, self.t, u,
                                       self.dt, self.params)
        self.step_count += 1
        
        #Compute reward
        q1 = self.state[:,0]
        q2 = self.state[:,1]
        u1 = self.state[:,2]
        u2 = self.state[:,3]

        q2 = jnp.mod(q2,2*jnp.pi) #normalize angle

        rewards = 10.0*jnp.cos(q2)
        - 0.1*q1**2
        - 0.1*u1**2
        - 0.1*u2**2

        #Check if terminated
        cart_out = jnp.abs(q1) > jnp.abs(self.action_bounds[1,1])
        max_steps_reached = self.step_count >= self.max_steps

        dones = cart_out | max_steps_reached

        info = {
            'cart_out': cart_out,
            'completed': max_steps_reached
        }

        return self.state, rewards, dones, info
    
    def render(self, env_idx=0):
        """
        Render a single environment from the batch
        
        Args:
            env_idx: which environment to visualize (default: first one)
        """
        # Extract state for this environment
        q1 = float(self.state[env_idx, 0])  # cart position (also fixed: was self.states)
        q2 = float(self.state[env_idx, 1])  # pole angle
        
        # Create figure only once (check if it exists)
        if not hasattr(self, 'fig') or not plt.fignum_exists(self.fig.number):
            self.fig, self.ax = plt.subplots(figsize=(8, 4))
        else:
            self.ax.clear()  # Clear previous frame
        
        # Track limits
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-0.5, 2)
        self.ax.set_aspect('equal')
        
        # Draw track
        self.ax.plot([-3, 3], [0, 0], 'k-', linewidth=2)
        
        # Draw cart (as rectangle)
        cart_width = 0.3
        cart_height = 0.2
        cart = patches.Rectangle(
            (q1 - cart_width/2, 0), 
            cart_width, 
            cart_height,
            facecolor='blue',
            edgecolor='black',
            linewidth=2
        )
        self.ax.add_patch(cart)
        
        # Draw pole (as line from cart center to pole tip)
        pole_length = 1.0  # visualize at 1.0 for clarity
        pole_x = q1 + pole_length * jnp.sin(q2)
        pole_y = cart_height + pole_length * jnp.cos(q2)
        
        self.ax.plot([q1, pole_x], [cart_height, pole_y], 'r-', linewidth=4)
        
        # Labels
        self.ax.set_xlabel('Position')
        self.ax.set_title(f'CartPole - Env {env_idx}\nAngle: {q2:.2f} rad, Position: {q1:.2f} m')
        self.ax.grid(True, alpha=0.3)
        
        plt.pause(0.01)  # Small pause for animation
        

if __name__ == "__main__":
    import jax.random as jr
    
    # Create environment
    env = CartPoleEnv(dt=0.02, max_steps=500, batch_size=4)
    
    # Reset
    states = env.reset()
    print(f"Initial states shape: {states.shape}")
    print(f"Initial states:\n{states}")
    
    # Run simulation with random actions
    plt.ion()  # Interactive plotting
    key = jr.PRNGKey(42)
    
    for step in range(200):
        # Random actions in [-1, 1]
        key, subkey = jr.split(key)
        actions = jr.uniform(subkey, (env.batch_size, 1), minval=-1.0, maxval=1.0)
        
        # Step environment
        next_states, rewards, dones, info = env.step(actions, max_force=100.0)
        
        # Print every 50 steps
        if step % 50 == 0:
            print(f"\nStep {step}:")
            print(f"  Time: {env.t:.3f} s")
            print(f"  Rewards: {rewards}")
            print(f"  Dones: {dones}")
            print(f"  Angle (env 0): {next_states[0, 1]:.3f} rad")
        
        # Render first environment
        env.render(env_idx=0)
        
        # Check if all done
        if jnp.all(dones):
            print(f"\nAll environments terminated at step {step}")
            break
    
    plt.ioff()
    plt.show()
    print("\nTest complete!")


