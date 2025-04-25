import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

# Lorenz system parameters
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0


# Lorenz system of equations
def lorenz(t, xyz):
    x, y, z = xyz
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]


# Initial conditions and time points
initial = [0.1, 0.0, 0.0]
t_span = (0, 50)
t_eval = np.linspace(t_span[0], t_span[1], 5000)

# Solve the system
solution = solve_ivp(lorenz, t_span, initial, t_eval=t_eval)
x, y, z = solution.y

# Set up the figure and 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Initialize the line object
line, = ax.plot([], [], [], 'b-', lw=0.5)
point, = ax.plot([], [], [], 'ro', markersize=5)
ax.set_xlim(min(x), max(x))
ax.set_ylim(min(y), max(y))
ax.set_zlim(min(z), max(z))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Lorenz Attractor Animation')


# Animation update function
def update(frame):
    # Update the line with all points up to current frame
    line.set_data(x[:frame], y[:frame])
    line.set_3d_properties(z[:frame])

    # Update the point at the current position
    point.set_data([x[frame]], [y[frame]])
    point.set_3d_properties([z[frame]])

    # Rotate the view slightly each frame
    ax.view_init(elev=20, azim=frame / 10)

    return line, point


# Create the animation
ani = FuncAnimation(fig, update, frames=len(x),
                    interval=20, blit=True)

plt.tight_layout()
plt.show()

# To save the animation (requires ffmpeg)
# ani.save('lorenz_attractor.mp4', writer='ffmpeg', fps=30, dpi=300)