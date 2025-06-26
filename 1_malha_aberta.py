import numpy as np
import matplotlib.pyplot as plt

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

# ------------------- 14715

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

print(np.linalg.eigvals(A))

B = np.array([[0], [0], [_E]]) # Command gain = E
C = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) # We only observe z´, which is x (car position)
D = np.array([[0]])

# Simulation parameters
t, tf, dt = 0, 50, .001 

# Start conditions
u, x = np.array([0]), np.array([[2], [0], [gravity*fluid_density*1]]) # Move car 3 meters, fill tank to 1 m

# Initialize arrays to store results
X, U, T = x, u, t

# Euler integration
dx = lambda x, u, dt: (np.dot(A,x)+np.dot(B,[u]))*dt

counter = 0
for i in range(int((tf-t)/dt)):
 #u = [.000001*counter]
 u = [0]
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
plt.legend()
plt.xlabel('Tempo (s)')
plt.ylabel('Posição mass | altura água (m)')
plt.grid(True)
plt.show()