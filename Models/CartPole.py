from sympy import symbols, lambdify, linear_eq_to_matrix
from sympy.physics.mechanics import *
import jax
import jax.numpy as jnp

# ------------------------
# Generalized coordinates
# ------------------------
q1, q2 = dynamicsymbols("q1 q2")  # cart position, pole angle
u1, u2 = dynamicsymbols("u1 u2")  # velocities
F = symbols("F")                  # input force
mc, mp, lp, Ip, g = symbols("mc mp lp Ip g")  # parameters

# ------------------------
# Frames and points
# ------------------------
ground_frame = ReferenceFrame("N")
ground_point = Point("O")

# Cart frame and center
cart_frame = ReferenceFrame("A")
cart_mass_point = Point("C")
cart_mass_point.set_vel(cart_frame, 0)  # initial vel = 0

# Pole frame and center
pole_frame = ReferenceFrame("B")
pole_mass_point = Point("P")

# ------------------------
# Bodies
# ------------------------
# Ground (small mass)
ground = RigidBody("ground", ground_point, ground_frame, 1e-6,
                   (inertia(ground_frame, 0, 0, 1e-6), ground_point))

# Cart as particle (1D mass)
cart = RigidBody("cart", cart_mass_point, cart_frame, mc,
                 (inertia(cart_frame, 0, 0, 1e-6), cart_mass_point))

# Pole as rigid body (mass + inertia)
pole = RigidBody("pole", pole_mass_point, pole_frame, mp,
                 (inertia(pole_frame, 0, 0, Ip), pole_mass_point))

# ------------------------
# Define joint points correctly
# ------------------------
# Cart origin (for prismatic joint)
cart_origin = cart.masscenter.locatenew("cart_origin", 0*cart_frame.x)

# Pole hinge at base of pole (relative to pole frame)
pole_base = pole.masscenter.locatenew("pole_base", -lp/2 * pole_frame.y)

# ------------------------
# Joints
# ------------------------
# Slider: Ground -> Cart
slider = PrismaticJoint(
    "slider",
    ground,
    cart,
    coordinates=q1,
    speeds=u1,
    child_interframe=cart_frame,
    parent_interframe=ground_frame,
)

# Revolute: Cart -> Pole
revolute = PinJoint(
    "revolute",
    cart,
    pole,
    coordinates=q2,
    speeds=u2,
    joint_axis=cart_frame.z,
    child_point=pole_base,
    parent_point=cart_origin,
    parent_interframe=cart_frame
)

# ------------------------
# Assemble system
# ------------------------
sys = System.from_newtonian(ground)
sys.add_bodies(cart, pole)
sys.add_joints(slider, revolute)

# ------------------------
# Loads
# ------------------------
sys.apply_uniform_gravity(-g * ground_frame.y)
sys.add_loads(Force(cart, F*ground_frame.x))

# ------------------------
# Equations of motion
# ------------------------
eom = sys.form_eoms(explicit_kinematics=True)

# Solve for accelerations
M, f = linear_eq_to_matrix(eom, [u1.diff(), u2.diff()])

# ------------------------
# Lambdify to jax
# ------------------------
M_func_jax = lambdify([q1, q2, u1, u2, mc, mp, lp, Ip], M, modules="jax")
f_func_jax = lambdify([q1, q2, u1, u2, F, mc, mp, lp, Ip, g], f, modules="jax")

def single_dynamics(xi, ui, params):
    q1, q2, u1, u2 = xi
    F_val = ui[0]
    mc_val, mp_val, lp_val, Ip_val, g_val = params

    M_val = M_func_jax(q1,q2,u1,u2,mc_val,mp_val,lp_val,Ip_val)
    f_val = f_func_jax(q1,q2,u1,u2,F_val,mc_val,mp_val,lp_val,Ip_val,g_val)

    qdd = jnp.linalg.solve(M_val, -f_val)
    return jnp.hstack([u1, u2, qdd[0], qdd[1]])

@jax.jit
def cartpole_dynamics_batched(x, u, params):
    return jax.vmap(single_dynamics, in_axes=(0,0,None))(x, u, params)

