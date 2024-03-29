import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
import pdb

def compute_curvature(x, y):
    # Compute derivatives of x and y
    dx = np.gradient(x)
    dy = np.gradient(y)

    # Compute second derivatives of x and y
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)

    # Compute curvature
    curvature = 1 - np.abs(d2x * dy - dx * d2y) / (dx**2 + dy**2)**(3/2)

    return curvature

def adaptive_sampling(x, y, curvature):
    # Define a sampling density function based on curvature
    scaling_factor = 1.0
    sampling_density = 1 - curvature * scaling_factor  # Adjust scaling_factor as needed

    num_points = len(x)
    sampled_indices = []

    # Add points based on sampling density
    for i in range(num_points):
        ran = np.random.rand()
        if ran < sampling_density[i]:
            sampled_indices.append(i)

    # Ensure the first and last points are included
    if 0 not in sampled_indices:
        sampled_indices.insert(0, 0)
    if num_points - 1 not in sampled_indices:
        sampled_indices.append(num_points - 1)

    return sampled_indices

# Example usage:
# Define a trajectory as a single long polyline segment
traj_x = np.linspace(0, 10, 1000)
traj_y = np.cos(traj_x)
# Calculate curvature
curvature = compute_curvature(traj_x, traj_y)
# Adaptive sampling
sampled_indices = adaptive_sampling(traj_x, traj_y, curvature)
#spline = CubicSpline(scaled_x, scaled_y)
spline_x = CubicSpline(range(len(traj_x)), traj_x)
spline_y = CubicSpline(range(len(traj_y)), traj_y)

spline_x_values = spline_x(sampled_indices)
spline_y_values = spline_y(sampled_indices)

# Plug into function
print(spline_x.c)
print(spline_y.c)
#coefficients = spline_func._spline.get_coeffs()
# Visualize the trajectory and ParamPoly3 curve
fig, ax = plt.subplots()

# Plot the trajectory
ax.plot(traj_x, traj_y, 'r-', label='Trajectory')
#ax.plot(scaled_x, scaled_y, 'bo', linewidth=2, label='Sampled Points')
ax.plot(spline_x_values, spline_y_values, 'b-', linewidth=2, label='Fitted spline')

ax.set_aspect('equal', 'box')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Trajectory and ParamPoly3 Curve Visualization')
plt.grid(True)
plt.legend()
plt.show()

