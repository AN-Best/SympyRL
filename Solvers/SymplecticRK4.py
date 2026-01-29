import jax
import jax.numpy as jnp

def symplectic_rk4_step(f, x, t, u, dt, params):
    """
    Batched 4th-order symplectic Runge-Kutta (2-stage Gauss-Legendre).
    
    Args:
        f: dynamics function f(x, u, params) -> dx/dt
        x: [batch, state_dim]
        t: scalar time
        u: [batch, action_dim]
        dt: timestep
        params: extra parameters for f
    Returns:
        x_next: [batch, state_dim]
        t_next: scalar
    """
    # Gauss-Legendre 2-stage coefficients
    c1 = 0.5 - jnp.sqrt(3)/6
    c2 = 0.5 + jnp.sqrt(3)/6
    a11 = 0.25
    a12 = 0.25 - jnp.sqrt(3)/6
    a21 = 0.25 + jnp.sqrt(3)/6
    a22 = 0.25
    b1 = 0.5
    b2 = 0.5

    # Initial guess for stages (explicit Euler)
    k1 = f(x, u, params)
    k2 = f(x, u, params)

    def stage_residual(k, _):
        # k = stacked [k1, k2], shape [batch, state_dim, 2]
        k1, k2 = k
        x1 = x + dt*(a11*k1 + a12*k2)
        x2 = x + dt*(a21*k1 + a22*k2)
        r1 = f(x1, u, params) - k1
        r2 = f(x2, u, params) - k2
        return (r1, r2), None

    # Fixed-point iteration for stages (5–10 iterations)
    k1, k2 = jax.lax.fori_loop(
        0, 5, lambda i, k: stage_residual(k, None)[0], (k1, k2)
    )

    # Combine stages for next state
    x_next = x + dt*(b1*k1 + b2*k2)
    t_next = t + dt
    return x_next, t_next
