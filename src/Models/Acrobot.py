from sympy import symbols, lambdify, linear_eq_to_matrix
from sympy.physics.mechanics import *
import jax
import jax.numpy as jnp
import torch
from sympy.printing.pycode import PythonCodePrinter

#Coordinates
q1, q2 = dynamicsymbols("q1 q2")  # angles
u1, u2 = dynamicsymbols("u1 u2")  # velocities
#Torque
T = dynamicsymbols("T") # torque
#Parameters
m, l, I, g = symbols("m l I g")  # parameters

#Frames
N = ReferenceFrame('N') #ground
A = ReferenceFrame('A') #pendulum 1
B = ReferenceFrame('B') #pendulum 2

#Orientation
A.orient(N,'Axis',[q1,N.z])
B.orient(A,'Axis',[q2,N.z])

#Angular velocity
A.set_ang_vel(N,u1*N.z)
B.set_ang_vel(A,u2*N.z)

#Points
O = Point('O')
O.set_vel(N,0)

P1 = O.locatenew('P1',(l/2)*A.x)
P2 = O.locatenew('P2',l*A.x + (l/2)*B.x)

P1.v2pt_theory(O,N,A)
P2.v2pt_theory(P1,N,B)


pendulum1 = RigidBody('pendulum1',P1,A,m,(inertia(A,0,0,I),P1))
pendulum2 = RigidBody('pendulum2',P2,B,m,(inertia(B,0,0,I),P2))

loads = [
    (P1, -m*g*N.y),
    (P2, -m*g*N.y),
    (A, -T*N.z),
    (B, T*N.z)
]

kd = [u1 - q1.diff(), u2 - q2.diff()]

KM = KanesMethod(N, q_ind=[q1, q2], u_ind=[u1, u2], kd_eqs=kd)
fr, frstar = KM.kanes_equations([pendulum1, pendulum2], loads)

eom = fr + frstar

M, f = linear_eq_to_matrix(eom, [u1.diff(), u2.diff()])

#Lambdify to JAX
M_func_jax = lambdify([q1, q2, u1, u2, m, l, I],M,modules="jax")
f_func_jax = lambdify([q1, q2, u1, u2, T, m, l, I, g],f,modules="jax")

#Lambdify to Torch
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
    [q1, q2, u1, u2, m, l, I],
    M,
    modules="torch",
    printer=torch_printer,
    cse = False)


f_func_torch = lambdify(
    [q1, q2, u1, u2, T, m, l, I, g],
    f,
    modules="torch",
    printer=torch_printer,
    cse = False)

#Jax dynamics function
def acrobot_dynamics_single_jax(xi, ui, params):
    q1, q2, u1, u2 = xi[0], xi[1], xi[2], xi[3]  
    T_val = ui[0]
    m_val, l_val, I_val, g_val = params

    M_val = M_func_jax(q1,q2,u1,u2,m_val,l_val,I_val)
    f_val = f_func_jax(q1,q2,u1,u2,T_val,m_val,l_val,I_val,g_val)

    qdd = jnp.linalg.solve(M_val, -f_val)
    return jnp.hstack([u1, u2, qdd[0], qdd[1]])

@jax.jit
def acrobot_dynamics_batched_jax(x, u, params):
    return jax.vmap(acrobot_dynamics_single_jax, in_axes=(0,0,None))(x, u, params)

#Torch dynamics function
def acrobot_dynamics_single_torch(xi, ui, params):
    q1, q2, u1, u2 = xi[0], xi[1], xi[2], xi[3]  
    T_val = ui[0]

    if not isinstance(params[0], torch.Tensor):
        # Convert to tensors matching the dtype/device of xi
        params = tuple(torch.as_tensor(p, dtype=xi.dtype, device=xi.device) for p in params)
    

    m_val, l_val, I_val, g_val = params

    M_val = M_func_torch(q1,q2,u1,u2,m_val,l_val,I_val)
    f_val = f_func_torch(q1,q2,u1,u2,T_val,m_val,l_val,I_val,g_val)

    qdd = torch.linalg.solve(M_val, -f_val)
    return torch.hstack([u1, u2, qdd[0], qdd[1]])

def acrobot_dynamics_batched_torch(x, u, params):
    return torch.vmap(acrobot_dynamics_single_torch, in_dims=(0, 0, None))(x, u, params)