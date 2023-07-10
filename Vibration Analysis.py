import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# Define system parameters
m = 1.0  # mass (kg)
k = 10.0  # stiffness (N/m)
c = 0.1  # damping coefficient (Ns/m)

# Formulate the equations of motion
def equations_of_motion(m, k, c):
    def fun(t, y):
        x, v = y
        dxdt = v
        dvdt = -(c*v + k*x) / m
        return [dxdt, dvdt]
    return fun

# Set up the time vector
t_start = 0.0
t_end = 10.0
dt = 0.01
t = np.arange(t_start, t_end, dt)

# Set up the initial conditions
x0 = 1.0  # initial displacement
v0 = 0.0  # initial velocity

# Solve the equations of motion using scipy's ODE solver
from scipy.integrate import solve_ivp

system = equations_of_motion(m, k, c)
sol = solve_ivp(system, [t_start, t_end], [x0, v0], t_eval=t)

# Extract the displacement and velocity from the solution
x = sol.y[0]
v = sol.y[1]

# Calculate the natural frequency and damping ratio
omega_n = np.sqrt(k/m)  # natural frequency (rad/s)
zeta = c / (2 * np.sqrt(m*k))  # damping ratio

# Print the natural frequency and damping ratio
print("Natural Frequency: {:.2f} rad/s".format(omega_n))
print("Damping Ratio: {:.2f}".format(zeta))

# Plot the displacement response
plt.figure()
plt.plot(t, x)
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.title('Displacement Response')
plt.grid(True)
plt.show()