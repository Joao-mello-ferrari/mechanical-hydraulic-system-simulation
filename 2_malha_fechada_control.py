import numpy as np
import matplotlib.pyplot as plt
from helpers.control import calculate_K

WATER_DENSITY = 10000

# Tank
fluid_density = WATER_DENSITY # kg/m^3
gravity = 9.81                # m/s^2
tank_area = 0.5               # m^2

# Pipe
fluid_density = WATER_DENSITY  # kg/m^3
pipe_length = 3                # m
pipe_section_area = 0.00125664 # m^2

# Mechanical system
mass = 5               # kg
spring_constant = 700   # N/m
damping_constant = 10  # Ns/m

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
C = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) # We only observe z´, which is x (car position)
D = np.array([[0]])

# --------------------

K = calculate_K(A, B)

# --------------------

# Simulation parameters
t, tf, dt = 0, 50, .001 

# Start conditions
u, x = np.array([0]), np.array([[-2], [0], [gravity*fluid_density*1]]) # Move car 3 meters, fill tank to 1 m

# Initialize arrays to store results
X, U, T = x, u, t

# Euler integration
dx = lambda x, u, dt: (np.dot(A,x)+np.dot(B,[u]))*dt

counter = 0
for i in range(int((tf-t)/dt)):
 u = -K@x
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