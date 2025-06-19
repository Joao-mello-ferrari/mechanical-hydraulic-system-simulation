import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def calculate_K(A, B, poles_gain, plot) -> np.ndarray:
  """
  Calculates the state feedback gain matrix K for pole placement in a state-space control system.

  This function computes the gain matrix K such that the closed-loop system (A - B*K) has its poles
  placed at desired locations in the complex plane. The desired pole locations are determined by
  shifting the original poles further left on the real axis (for increased stability) and adjusting
  their imaginary parts, both controlled by the `poles_gain` parameter.

  Args:
    A (np.ndarray): The state matrix of the system (n x n).
    B (np.ndarray): The input matrix of the system (n x m).
    poles_gain (float): Gain factor to adjust the placement of the new poles.
    plot (bool): If True, plots the original and new pole locations on the complex plane.

  Returns:
    np.ndarray: The computed state feedback gain matrix K (1 x n).

  Notes:
    - The function uses symbolic computation to solve for K such that the characteristic polynomial
      of (A - B*K) matches the desired polynomial with new pole locations.
    - The function assumes a 3-state system (n=3) and a single input (m=1).
    - If `plot` is True, a plot of the original and new poles is displayed.
  """
  # Calculate new poles
  poles = np.linalg.eigvals(A)

  # Set new poles to be more closer to infinity on real axis and close to zero on imaginary axis
  new_x, new_y = -abs(np.mean(np.real(poles))) * poles_gain, abs(np.mean(np.imag(poles))) / poles_gain

  # det(sI - A + BK) = (s - p1)(s - p2)(s - p3)
  s = sp.symbols('s')

  k1, k2, k3 = sp.symbols('k1 k2 k3')
  K = sp.Matrix([[k1, k2, k3]])

  # This creates a polynome with the s as symbolic variable 
  # and the coefficients as k1, k2, k3
  left_side_poly = (sp.eye(3) * s - A + B@K).det().simplify()

  # This creates a polynomial with the s as symbolic variable
  # and the coefficients as new_x, new_y, out aimed poles
  # p1 = p2 = p3 = new_x + new_y*j
  p1, p2, p3 = new_x + new_y * 1j, new_x + new_y * 1j, new_x + new_y * 1j
  right_side_poly = sp.expand((s - p1) * (s - p2) * (s - p3))

  # Create Polynomials objects and match coefficients
  left_side_poly_coeffs = sp.Poly(left_side_poly, s).all_coeffs()
  right_side_poly_coeffs = sp.Poly(right_side_poly, s).all_coeffs()
  eqs = [sp.Eq(c1, c2) for c1, c2 in zip(left_side_poly_coeffs, right_side_poly_coeffs)]

  # Compute Ks
  solution = sp.solve(eqs, (k1, k2, k3))
  K = solution[k1], solution[k2], solution[k3]
  print("\n\033[95mSolution for K:\033[0m")
  print(K)

  if plot:
    plt.plot(np.real(poles),np.imag(poles),'bx', label='Original poles')
    plt.plot(np.real([p1, p2, p3]),np.imag([k1, k2, k3]),'rx', label='New poles')
    plt.xlabel('Real axis')
    plt.ylabel('Imaginary axis')
    plt.grid(True)
    plt.legend()
    plt.title('Pole Placement')
    plt.show()

  return np.array(K, dtype=float)