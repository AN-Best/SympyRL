import jax
import jax.numpy as jnp

def implicit_midpoint_newton(f, x, t, u, dt, params, max_iter=10):
    """
    Batched implicit midpoint integrator using Newton-Raphson.

    Args:
        f: dynamics function f(x, u, params) -> dx/dt, batched
        x: [batch, state_dim] current state
        t: scalar time
        u: [batch, action_dim] control input
        dt: timestep
        params: extra parameters to pass to f
        max_iter: Newton-Raphson iterations
    Returns:
        x_next: [batch, state_dim] next state
        t_next: scalar
    """

    # Initial guess: explicit Euler
    x_next_init = x + dt * f(x, u, params)

    carry_init = (x, x_next_init)  # (x_prev, x_next_curr)

    def body_fun(carry, _):
        x_prev, x_next_curr = carry

        # Midpoint
        xm = 0.5 * (x_prev + x_next_curr)

        # Residual
        r = x_next_curr - x_prev - dt * f(xm, u, params)

        # Flatten for Jacobian solve
        def r_flat(xn_flat):
            xn = xn_flat.reshape(x_next_curr.shape)
            xm = 0.5 * (x_prev + xn)
            r = xn - x_prev - dt * f(xm, u, params)
            return r.ravel()

        r_val = r_flat(x_next_curr.ravel())
        J = jax.jacobian(r_flat)(x_next_curr.ravel())

        # Newton update
        dx = jnp.linalg.solve(J, -r_val)
        x_next_new = (x_next_curr.ravel() + dx).reshape(x_next_curr.shape)

        # Return new carry, y placeholder (None)
        return (x_prev, x_next_new), None

    # Run Newton-Raphson iterations
    (_, x_next_final), _ = jax.lax.scan(body_fun, carry_init, None, length=max_iter)

    t_next = t + dt
    return x_next_final, t_next
