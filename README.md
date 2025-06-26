## Demo
To access the simulation demo, please follow
https://youtu.be/bVrPHuKIi-w
![aberta](https://github.com/user-attachments/assets/f143c186-ffd3-43f0-929a-7a70ca6416f5)


## System Overview

- **Mechanical Subsystem:** A mass attached to a spring and damper, representing a classic mass-spring-damper system.
- **Hydraulic Subsystem:** A tank connected to the mass via a pipe, where the pressure in the tank is affected by the mass position and fluid flow.
- **Coupling:** The force on the mass is coupled to the pressure in the tank through the piston and pipe.

---

## Control Approach

- **State-Space Modeling:** The system is modeled using state-space equations, capturing the dynamics of mass position, velocity, and tank pressure.
- **Controllability & Observability:** The code checks if the system can be fully controlled and observed from the chosen inputs and outputs.
- **State Feedback Control:** A feedback gain (K) is computed using pole placement to ensure desired closed-loop dynamics.
- **Reference Tracking:** A reference gain (N) is calculated to allow the system to track a desired mass position.
- **State Estimation:** A Luenberger observer (gain L) is designed to estimate the full system state from partial measurements (mass position).

---

## Simulation

- **Numerical Integration:** The system and observer are simulated over time using Euler integration.
- **Visualization:** The results include plots of:
  - Actual vs. estimated mass position
  - Mass speed
  - Tank pressure (or fluid height)
  - Control input (flow rate `J`)

---

## Key Features

- Demonstrates modern control techniques, including:
  - State feedback control
  - Observer (state estimation) design
  - Reference tracking
- Provides insight into mechanical-hydraulic coupled systems.
- Helps visualize controller and observer performance.

---

## 🚀 How to Run

1. Install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Run the simulation:

```bash
make run
```
