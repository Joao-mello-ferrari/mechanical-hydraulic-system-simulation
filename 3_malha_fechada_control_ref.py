import numpy as np
import matplotlib.pyplot as plt
from helpers.control import calculate_K
from helpers.reference import calculate_N

WATER_DENSITY = 1000
MERCURY_DENSITY = 13546

# Tank
fluid_density = MERCURY_DENSITY # kg/m^3
gravity = 9.81                  # m/s^2
tank_area = 0.05                # m^2 (0.005 is a 7cm x 7cm square tank)

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
C = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) # We observe all states: mass position, mass speed, and tank pressure
D = np.array([[0]])

# --------------------

K = calculate_K(A, B, poles_gain=10, plot=False)

# --------------------

N = calculate_N(A, B, C[:1], D, K, A.shape[1], B.shape[1])

# --------------------

# Simulation parameters
t, tf, dt = 0, 50, .001 

# Start conditions. We assume there is someone only pulling/pushing the spring.
# The tank height should reflect that mass position: it depends on.
initial_mass_position = 2
initial_tank_height = (spring_constant*initial_mass_position) / (fluid_density*gravity*piston_area)

# Set start conditions
u = np.array([1]) # No water flow at the beginning
x = np.array([[initial_mass_position], [0], [gravity*fluid_density*initial_tank_height]]) # Move car 3 meters, fill tank to 1 m
r = np.array([[5]]) # Set 5 meters to be mass position reference

# Initialize arrays to store results
X, U, T = x, u, t

# Euler integration
dx = lambda x, u, dt: (np.dot(A,x)+np.dot(B,u))*dt

counter = 0
for i in range(int((tf-t)/dt)):
 u = N@r - K@x
 #print(u)
 #u = [0]
 t, x = t + dt, x + dx(x,u,dt)
 X = np.append(X,x,axis=1)
 U = np.append(U,u)
 T = np.append(T,t)

 counter += 1

output = np.dot(C,X)
mass_position = output[0]
mass_speed = output[1]
tank_height = output[2] / (gravity*fluid_density)
print(mass_position[-1], mass_speed[-1], tank_height[-1])

plt.plot(T,mass_position,'k', label='Posição da massa (m)')
plt.plot(T,mass_speed,'r', label='Velocidade da massa (m/s)')
plt.plot(T,tank_height,'b', label='Altura da água no tanque (m)')
plt.plot(T,U,'y', label='Vazão (m^3/s)')
plt.legend()
plt.xlabel('Tempo (s)')
plt.ylabel('Posição mass | altura água (m)')
plt.grid(True)
plt.show()