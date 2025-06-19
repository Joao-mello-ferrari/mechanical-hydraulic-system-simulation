import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from helpers.conditions import check_controllability, check_observability
from helpers.control import calculate_K
from helpers.reference import calculate_N
from helpers.estimator import calculate_L

WATER_DENSITY = 1000
MERCURY_DENSITY = 13546

# Tank
fluid_density = MERCURY_DENSITY # kg/m^3
gravity = 9.81                # m/s^2
tank_area = 0.05              # m^2 (0.005 is a 7cm x 7cm square tank)

# Pipe
fluid_density = MERCURY_DENSITY  # kg/m^3
pipe_length = 10                # m
pipe_section_area = 0.00125664 # m^2

# Mechanical system
mass = 5               # kg
spring_constant = 500   # N/m
damping_constant = 40  # Ns/m

# Coupling system
piston_area = pipe_section_area

# -------------------

# Ax´´ + Bx´+ Cx - P1 = 0
# P1´ = Dx´+ EJ
_A = (mass / piston_area) + (fluid_density*pipe_length*piston_area / pipe_section_area)
_B = damping_constant / piston_area
_C = spring_constant / piston_area
_D = piston_area*fluid_density*gravity / tank_area
_E = fluid_density*gravity / tank_area

A = np.array([
    [ 0,       1,       0  ],
    [ -_C/_A,  -_B/_A,  1/_A ],
    [ 0,       -_D,     0  ]
])
B = np.array([[0], [0], [_E]]) # Command gain = E
C = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) # We can observe all states: mass position, mass speed, and tank pressure
D = np.array([[0]])

# --------------------

controllable = check_controllability(A, B)
observable = check_observability(A, C)
print("\033[95mControllable:\033[0m", controllable)
print("\033[95mObservable:\033[0m", observable)

# --------------------

K = calculate_K(A, B, poles_gain=10, plot=True)

# --------------------

N = calculate_N(A, B, C[:1], D, K, A.shape[1], B.shape[1])

# --------------------

L = calculate_L(A, C[:1], poles_gain=20, plot=True)

# --------------------

# Simulation parameters
t, tf, dt = 0, 50, .001 

# Start conditions. We assume there is someone only pulling/pushing the spring.
# The tank height should reflect that mass position
# We get the initial tank height from: P = g*p*h, being P = f/A = k*x/A
initial_mass_position = 2
initial_tank_height = (spring_constant*initial_mass_position) / (fluid_density*gravity*piston_area)

# Set start conditions
u = np.array([0]) # No water flow at the beginning
x = np.array([[initial_mass_position], [0], [0]])
x_est = np.array([[initial_mass_position], [0], [0]])
r = np.array([[3]]) # Set 3 meters to be mass position reference

# Initialize arrays to store results
X, U, T, X_est = x, u, t, x_est

# Euler integration
dx = lambda x, u, dt: (np.dot(A,x)+np.dot(B,u))*dt
dx_est = lambda x_est, y, y_est, u, dt: (np.dot(A,x_est)+np.dot(B,u)+L@(y-y_est))*dt

num_steps = int((tf-t)/dt)
for i in trange(num_steps, desc="Simulating"):
    # Calculate control from x_est and reference
    u = N@r - K@x_est

    # We use C[:1] since we observe only the first state (mass position)
    y, y_est = C[:1]@x, C[:1]@x_est

    # Integrate the system and the estimator by euler method
    t, x = t + dt, x + dx(x,u,dt)
    x_est = x_est + dx_est(x_est, y, y_est, u,dt)

    X = np.append(X,x,axis=1)
    X_est = np.append(X_est,x_est,axis=1)
    U = np.append(U,u)
    T = np.append(T,t)

output = np.dot(C,X)
mass_position = output[0]
mass_speed = output[1]
tank_height = output[2] / (gravity*fluid_density)
print("\n\033[96mFinal x values:\033[0m")
print(mass_position[-1], mass_speed[-1], tank_height[-1])

est_output = np.dot(C,X_est)
mass_position_est = est_output[0]
mass_speed_est = est_output[1]
tank_height_est = est_output[2] / (gravity*fluid_density)
print("\n\033[96mFinal x_estimated values:\033[0m")
print(mass_position_est[-1], mass_speed_est[-1], tank_height_est[-1])

plt.plot(T, mass_position, color='black', label='Mass position (m)')
plt.plot(T, mass_position_est, color='green', label='Estimated mass position (m)')
plt.plot(T, mass_speed, color='red', label='Mass speed (m/s)')
plt.plot(T, mass_speed_est, color='magenta', label='Estimated mass speed (m/s)')
plt.plot(T, tank_height, color='blue', label='Tank water height (m)')
plt.plot(T, U, 'y', label='Flow rate (m^3/s)')
plt.axhline(y=r[0,0], color='gray', linestyle='--', label=f'Mass reference position ({r[0,0]} m)')
plt.title('Closed-loop control with state estimation')
plt.ylabel('Control variables in SI units')
plt.xlabel('Time (s)')
plt.legend()
plt.grid(True)
plt.show()