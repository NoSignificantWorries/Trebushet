import numpy as np
import matplotlib
from scipy.integrate import solve_ivp
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class const:
    # trebushet params
    h = 0.16
    l1, l2 = 0.055, 0.15
    l3 = (l1 + l2) / 2
    phi_start = np.pi / 4
    phi_end = np.pi
    r = 0.012
    mg = 80 / 1000
    ms = 5 / 1000
    mr = 3 / 1000
    Jg = 0.4 * mg * 0.03 ** 2

    # windage
    S = np.pi * r ** 2
    C = 0.47
    rho = 1.225

    # target zone
    d1, d2 = 0.5, 0.8

    # gravity constant
    g = 9.81


# shell's inertia
Js = const.ms * const.l2 ** 2
# lever's inertia
Jr = const.mr * (const.l1 + const.l2) ** 2 / 3
# load's inertia
Jg = const.Jg + const.mg * const.l1 ** 2
# common inertia
J = Js + Jr + Jg
print("Common inertia:", J, "kg*m^2")

# constant of moment
M0 = const.g * (const.mg * const.l1 - const.ms * const.l2 - const.mr * const.l3)
print("Constant of moment", M0, "H")

# rotation speed by the angle
print(2 * M0 / J * (np.sin(const.phi_end) - np.sin(const.phi_start)))
w = np.sqrt(2 * M0 / J * (np.cos(const.phi_start) - np.cos(const.phi_end)))
print("Rotation speed:", w, "rad/s")

# vector of velocity by the coordinates
voy = const.l2 * np.sin(const.phi_end) * w
vox = -const.l2 * np.cos(const.phi_end) * w
print("Velosity X:", vox, "m/s")
print("Velocity Y:", voy, "m/s")

# state of rest position of the lever
A0 = (-const.l2 * np.sin(const.phi_start), const.h - const.l2 * np.cos(const.phi_start))
B0 = (const.l1 * np.sin(const.phi_start), const.h + const.l1 * np.cos(const.phi_start))

# start position of the level
A = (-const.l2 * np.sin(const.phi_end), const.h - const.l2 * np.cos(const.phi_end))
x0, y0 = A
B = (const.l1 * np.sin(const.phi_end), const.h + const.l1 * np.cos(const.phi_end))


def f(x):
    '''
    Trajectory of shell's movement
    '''
    return y0 + voy / vox * (x - x0) - const.g * (x - x0) ** 2 / (2 * vox ** 2)


# calculating shot distance
D = 4 * (vox * voy + const.g * x0) ** 2 + 4 * const.g * (2 * vox ** 2 * (y0 - const.r) - x0 * (2 * vox * voy + const.g * x0))
L = (2 * (vox * voy + const.g * x0) + np.sqrt(D)) / (2 * const.g)
print("Discriminant:", D)
print("Distance", L, "m")
Ly = f(L)


def projectile_motion(t, state, m, g, rho, C_d, A):
    '''
    Equation of motion with air resistance
    '''
    _, vx, _, vy = state
    v = np.sqrt(vx ** 2 + vy ** 2)
    
    # power of air resistance
    F_d = 0.5 * rho * C_d * A * v ** 2
    
    dxdt = vx
    dvxdt = - (F_d / m) * (vx / v) if v > 0 else 0
    dydt = vy
    dvydt = -g - (F_d / m) * (vy / v) if v > 0 else -g
    
    return [dxdt, dvxdt, dydt, dvydt]


# intial state of movement
initial_state = [A[0], vox, A[1], voy]

# 10 seconds to move
t_span = (0, 10)
t_eval = np.linspace(0, 10, 1000)

# solving the equation
solution = solve_ivp(
    projectile_motion,
    t_span,
    initial_state,
    args=(const.ms, const.g, const.rho, const.C, const.S),
    t_eval=t_eval,
    method='RK45',
    rtol=1e-6,
    atol=1e-9
)

# result trajectory with air resistance
x = solution.y[0]
y = solution.y[2]

# cropping to correct height
ground_idx = np.argwhere(y <= 0).flatten()[0]
x = x[:ground_idx]
y = y[:ground_idx]
print("Distance with air resistance:", x[-1], "m")


# plotting
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

# target zone
ax.axvspan(const.d1, const.d2, alpha=0.3, color='#cba6f7', label="Target zone")

# trajectories
X = np.linspace(x0, L,  300)
ax.plot(X, f(X), "g", linewidth=2, label="Trajectory of movement")
ax.plot(x, y, "r--", linewidth=1, alpha=0.7, label="Real trajectory of movement")

# lever
ax.plot([A0[0], B0[0]], [A0[1], B0[1]], "r--o", alpha=0.4, label="Start position")
ax.plot([A[0], B[0]], [A[1], B[1]], "r-o", label="Shot position")

# final position
ax.axvline(L, color="k", linestyle="--", label=f"Final distance = {L:.4f}m")
ax.axvline(x[-1], color="k", linestyle="--", linewidth=1, label=f"Final real distance = {x[-1]:.4f}m")

# adding circles
circle_end = plt.Circle((L, Ly), radius=const.r, color='green', alpha=0.5, fill=True, label="Final shell's position")
circle_real = plt.Circle((x[-1], y[-1]), radius=const.r, color='red', alpha=0.3, fill=True, label="Final shell's real position")
circle_start = plt.Circle(A, radius=const.r, linestyle="--", color='blue', alpha=0.3, fill=True)
circle0 = plt.Circle(A0, radius=const.r, linestyle="--", color='blue', alpha=0.2, fill=True)
ax.add_patch(circle_end)
ax.add_patch(circle_real)
ax.add_patch(circle_start)
ax.add_patch(circle0)

ax.axis("scaled")
ax.set_ylim(bottom=0)
ax.legend()

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("trajectory.png", dpi=200)

