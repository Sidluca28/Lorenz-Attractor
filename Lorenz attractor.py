import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sigma = 10
rho = 28
beta = 8/3

def lorenz(t, state):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

t_span = (0, 40)
t_eval = np.linspace(t_span[0], t_span[1], 10000)
initial_state = [1.0, 1.0, 1.0]

sol = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(sol.y[0], sol.y[1], sol.y[2], color='blue', lw=0.5)
ax.set_title("Lorenz Attractor", fontsize=16)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axs[0].plot(sol.t, sol.y[0], label="x(t)", color='red')
axs[0].set_ylabel("x(t)")
axs[1].plot(sol.t, sol.y[1], label="y(t)", color='green')
axs[1].set_ylabel("y(t)")
axs[2].plot(sol.t, sol.y[2], label="z(t)", color='blue')
axs[2].set_ylabel("z(t)")
axs[2].set_xlabel("Time")

fig.suptitle("Time Series of Lorenz System", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
