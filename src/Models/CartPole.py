from sympy import symbols, lambdify, linear_eq_to_matrix
from sympy.physics.mechanics import *
import jax
import jax.numpy as jnp
import torch
from sympy.printing.pycode import PythonCodePrinter

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

# ------------------------
# Lambdify to torch
# ------------------------

class TorchPrinter(PythonCodePrinter):

    def __init__(self):
        super().__init__()
        # Map function names to torch equivalents
        self.known_functions = {
            'sin': 'torch.sin',
            'cos': 'torch.cos',
            'tan': 'torch.tan',
            'asin': 'torch.asin',
            'acos': 'torch.acos',
            'atan': 'torch.atan',
            'atan2': 'torch.atan2',
            'exp': 'torch.exp',
            'log': 'torch.log',
            'sqrt': 'torch.sqrt',
            'abs': 'torch.abs',
        }

    """Custom printer that uses ** operator instead of pow()"""
    
    def _print_Pow(self, expr):
        base = self._print(expr.base)
        exp = self._print(expr.exp)
        return f"({base})**({exp})"
    
    
    def _print_ImmutableDenseMatrix(self, expr):
        """Print matrix as nested torch.stack calls"""
        rows = []
        for i in range(expr.rows):
            row_elements = [self._print(expr[i, j]) for j in range(expr.cols)]
            if expr.cols == 1:
                # Single column - just the element
                rows.append(row_elements[0])
            else:
                # Multiple columns - stack along last dimension
                rows.append(f"torch.stack([{', '.join(row_elements)}], dim=-1)")
        
        if expr.rows == 1:
            # Single row
            if expr.cols == 1:
                return rows[0]
            else:
                return rows[0]  # Already stacked above
        else:
            # Multiple rows
            if expr.cols == 1:
                # Vector: stack elements (no extra dim needed)
                return f"torch.stack([{', '.join(rows)}], dim=-1)"
            else:
                # Matrix: stack rows
                return f"torch.stack([{', '.join(rows)}], dim=-1)"

# Use the custom printer
torch_printer = TorchPrinter()

M_func_torch = lambdify(
    [q1, q2, u1, u2, mc, mp, lp, Ip],
    M,
    modules='torch',
    printer=torch_printer,
    cse=False
)

f_func_torch = lambdify(
    [q1, q2, u1, u2, F, mc, mp, lp, Ip, g],
    f,
    modules='torch',
    printer=torch_printer,
    cse=False
)

# ------------------------
# Jax Dynamics Function
# ------------------------
def cartpole_dynamics_single_jax(xi, ui, params):
    q1, q2, u1, u2 = xi[0], xi[1], xi[2], xi[3]  
    F_val = ui[0]
    mc_val, mp_val, lp_val, Ip_val, g_val = params

    M_val = M_func_jax(q1,q2,u1,u2,mc_val,mp_val,lp_val,Ip_val)
    f_val = f_func_jax(q1,q2,u1,u2,F_val,mc_val,mp_val,lp_val,Ip_val,g_val)

    qdd = jnp.linalg.solve(M_val, -f_val)
    return jnp.hstack([u1, u2, qdd[0], qdd[1]])

@jax.jit
def cartpole_dynamics_batched_jax(x, u, params):
    return jax.vmap(cartpole_dynamics_single_jax, in_axes=(0,0,None))(x, u, params)

# ------------------------
# Torch Dynamics Function
# ------------------------
def cartpole_dynamics_single_torch(xi, ui, params):
    q1, q2, u1, u2 = xi[0], xi[1], xi[2], xi[3]  
    F_val = ui[0]

    if not isinstance(params[0], torch.Tensor):
        # Convert to tensors matching the dtype/device of xi
        params = tuple(torch.as_tensor(p, dtype=xi.dtype, device=xi.device) for p in params)
    
    mc_val, mp_val, lp_val, Ip_val, g_val = params

    M_val = M_func_torch(q1,q2,u1,u2,mc_val,mp_val,lp_val,Ip_val)
    f_val = f_func_torch(q1,q2,u1,u2,F_val,mc_val,mp_val,lp_val,Ip_val,g_val)

    qdd = torch.linalg.solve(M_val, -f_val)
    return torch.hstack([u1, u2, qdd[0], qdd[1]])

def cartpole_dynamics_batched_torch(x, u, params):
    return torch.vmap(cartpole_dynamics_single_torch, in_dims=(0, 0, None))(x, u, params)

