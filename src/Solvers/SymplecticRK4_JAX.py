import jax
import jax.numpy as jnp

# Precompute Gauss-Legendre 2-stage coefficients
c1 = 0.5 - jnp.sqrt(3)/6
c2 = 0.5 + jnp.sqrt(3)/6
a11 = 0.25
a12 = 0.25 - jnp.sqrt(3)/6
a21 = 0.25 + jnp.sqrt(3)/6
a22 = 0.25
b1 = 0.5
b2 = 0.5

def symplectic_rk4_step_jax(f, x, t, u, dt, params, iterations=5):
    """
    Batched 4th-order symplectic Runge-Kutta (2-stage Gauss-Legendre).
    
    f: dynamics function f(x, u, params) -> dx/dt, returns [batch,state_dim]
    x: [batch, state_dim]
    t: scalar
    u: [batch, action_dim]
    dt: timestep
    params: extra parameters for f
    iterations: fixed-point iterations for implicit stages
    """
    # Initial guess for stages (explicit Euler)
    k = jnp.stack([f(x, u, params), f(x, u, params)], axis=-1)  # shape [batch, state_dim, 2]

    def stage_iter(i, k):
        # k[:,:,0] = k1, k[:,:,1] = k2
        x1 = x + dt*(a11*k[:,:,0] + a12*k[:,:,1])
        x2 = x + dt*(a21*k[:,:,0] + a22*k[:,:,1])
        k1_new = f(x1, u, params)
        k2_new = f(x2, u, params)
        return jnp.stack([k1_new, k2_new], axis=-1)

    k = jax.lax.fori_loop(0, iterations, stage_iter, k)

    # Combine stages for next state
    x_next = x + dt*(b1*k[:,:,0] + b2*k[:,:,1])
    t_next = t + dt
    return x_next, t_next
