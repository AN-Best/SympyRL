def rk4_step(f, x, t, u, dt,*f_args):
    """
    Args:
        f: dynamics function f(x, u) -> dx/dt, batched or unbatched
        x: [batch, state_dim] state
        t: scalar time
        u: [batch, action_dim] control input
        dt: scalar timestep
    Returns:
        x_next: [batch, state_dim]
        t_next: scalar
    """
    k1 = f(x, u, *f_args)
    k2 = f(x + 0.5 * dt * k1, u, *f_args)
    k3 = f(x + 0.5 * dt * k2, u, *f_args)
    k4 = f(x + dt * k3, u, *f_args)

    x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    t_next = t + dt
    return x_next, t_next

