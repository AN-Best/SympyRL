from sympy import symbols, lambdify, solve
from sympy.physics.mechanics import *
import torch

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

accelerations = solve(eom, [u1.diff(), u2.diff()], dict=True)[0]

u1d_func_torch = lambdify(
    [q1, q2, u1, u2, T, m, l, I, g],
    accelerations[u1.diff()],
    modules="torch"
)
u2d_func_torch = lambdify(
    [q1, q2, u1, u2, T, m, l, I, g],
    accelerations[u2.diff()],
    modules="torch"
)

# ------------------------
# Torch batched dynamics
# ------------------------
class AcrobotDynamicsBatched(torch.nn.Module):
    def __init__(self, params):
        """
        params: list or tensor [m, l, I, g]
        Should be torch tensor on the correct device for GPU use.
        """
        super().__init__()
        # Convert scalars to tensors once, on correct device
        if not isinstance(params, torch.Tensor):
            params = torch.tensor(params, dtype=torch.float32)
        self.register_buffer("params", params)  # registers a buffer, so it moves with .to(device)

    def forward(self, x, u):
        """
        x: [batch, 4] -> [q1, q2, u1, u2]
        u: [batch, 1] -> [T]
        returns: [batch, 4] -> [q1', q2', u1', u2']
        """
        q1, q2, u1, u2 = x[:,0], x[:,1], x[:,2], x[:,3]
        F_val = u[:,0]

        m_val, l_val, I_val, g_val = self.params

        # Call torch-lambdified functions
        q1d = u1d_func_torch(q1, q2, u1, u2, F_val, m_val, l_val, I_val, g_val)
        q2d = u2d_func_torch(q1, q2, u1, u2, F_val, m_val, l_val, I_val, g_val)

        # Clamp small numerical instabilities
        q1d = torch.nan_to_num(q1d, nan=0.0, posinf=50.0, neginf=-50.0)
        q2d = torch.nan_to_num(q2d, nan=0.0, posinf=50.0, neginf=-50.0)

        return torch.stack([u1, u2, q1d, q2d], dim=1)



