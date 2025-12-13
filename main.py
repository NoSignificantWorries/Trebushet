import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class const:
    l1, l2 = 0.055, 0.15
    l3 = (l1 + l2) / 2
    phi_start = -4 * np.pi / 5
    phi_end = 2
    r = 0.012
    h = 0.17
    d1, d2 = 0.5, 0.8
    g = 9.81
    mg = 80 / 1000
    ms = 5 / 1000
    mr = 3 / 1000
    Jg = 0.4 * mg * 0.03 ** 2


Js = const.ms * const.l2 ** 2
Jr = const.mr * (const.l1 + const.l2) ** 2 / 12 + const.mr * const.l3 ** 2
Jg = const.Jg + const.mg * const.l1 ** 2
J = Js + Jr + Jg
print(J)

M0 = const.g * (const.mg * const.l1 - const.ms * const.l2 - const.mr * const.l3)
print(M0)

w = np.sqrt(2 * M0 / J * (np.sin(const.phi_end) - np.sin(const.phi_start)))
print(w)

vox = -const.l2 * np.sin(const.phi_end) * w
voy = const.l2 * np.cos(const.phi_end) * w
print(vox)
print(voy)

A0 = (const.l2 * np.cos(const.phi_start), const.h + const.l2 * np.sin(const.phi_start))
B0 = (-const.l1 * np.cos(const.phi_start), const.h - const.l1 * np.sin(const.phi_start))

A = (const.l2 * np.cos(const.phi_end), const.h + const.l2 * np.sin(const.phi_end))
x0, y0 = A
B = (-const.l1 * np.cos(const.phi_end), const.h - const.l1 * np.sin(const.phi_end))

def f(x):
    return y0 + voy / vox * (x - x0) - const.g * (x - x0) ** 2 / (2 * vox ** 2)

D = 4 * (vox * voy + const.g * x0) ** 2 + 4 * const.g * (2 * vox ** 2 * (y0 - const.r) - x0 * (2 * vox * voy + const.g * x0))
L = (2 * (vox * voy + const.g * x0) + np.sqrt(D)) / (2 * const.g)
print(D)
print(L)
Ly = f(L)
print(Ly)


fig, ax = plt.subplots(1, 1, figsize=(10, 10))

ax.axvspan(const.d1, const.d2, alpha=0.3, color='#cba6f7')

X = np.linspace(x0, L,  300)

ax.plot([A0[0], B0[0]], [A0[1], B0[1]], "r--o", alpha=0.4)
ax.plot([A[0], B[0]], [A[1], B[1]], "r-o")
ax.plot(X, f(X))
ax.axvline(L, color="k", linestyle="--")
circle_end = plt.Circle((L, Ly), radius=const.r, color='blue', alpha=0.5, fill=True)
ax.add_patch(circle_end)
circle_start = plt.Circle(A, radius=const.r, color='blue', alpha=0.5, fill=True)
ax.add_patch(circle_start)
circle0 = plt.Circle(A0, radius=const.r, color='blue', alpha=0.5, fill=True)
ax.add_patch(circle0)

ax.axis("scaled")
ax.set_ylim(bottom=0)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("res.png", dpi=200)

