import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Lorenz system parameters
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

# Lorenz equations
def lorenz(t, state):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

# Time configuration
t_span = (0, 40)
t_eval = np.linspace(t_span[0], t_span[1], 10000)

# Initial conditions
init_1 = [1.0, 1.0, 1.0]
init_2 = [1.001, 1.0, 1.0]  # slightly perturbed to show sensitivity

# Solve both trajectories
sol1 = solve_ivp(lorenz, t_span, init_1, t_eval=t_eval)
sol2 = solve_ivp(lorenz, t_span, init_2, t_eval=t_eval)

# Plotting the divergence of trajectories
plt.figure(figsize=(10, 6))
plt.plot(t_eval, np.abs(sol1.y[0] - sol2.y[0]), label='|x1 - x2|')
plt.plot(t_eval, np.abs(sol1.y[1] - sol2.y[1]), label='|y1 - y2|')
plt.plot(t_eval, np.abs(sol1.y[2] - sol2.y[2]), label='|z1 - z2|')
plt.yscale('log')
plt.xlabel("Time")
plt.ylabel("Absolute Difference (log scale)")
plt.title("Sensitive Dependence on Initial Conditions")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2D projection: x vs z
plt.figure(figsize=(8, 6))
plt.plot(sol1.y[0], sol1.y[2], color='purple')
plt.xlabel("x")
plt.ylabel("z")
plt.title("Lorenz Attractor: x vs z Projection")
plt.grid(True)
plt.tight_layout()
plt.show()


