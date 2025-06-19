import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def calculate_L(A, C, poles_gain, plot) -> np.ndarray:
  """
  Calculates the observer gain matrix L for a given system using pole placement.

  This function computes the observer gain matrix L such that the eigenvalues (poles) of the observer error dynamics
  (A - LC) are placed at desired locations in the complex plane. The desired poles are determined by shifting the 
  original system poles further into the left half-plane (for stability) and adjusting their imaginary parts, 
  controlled by the `poles_gain` parameter.

  Args:
    A (np.ndarray): The state matrix of the system (n x n).
    C (np.ndarray): The output matrix of the system (1 x n or m x n).
    poles_gain (float): Gain factor to adjust the location of the desired poles.
    plot (bool): If True, plots the original and new pole locations on the complex plane.

  Returns:
    np.ndarray: The observer gain matrix L (n x 1).

  Notes:
    - The function assumes a 3-state system (n=3).
    - Symbolic computation is used to solve for L.
    - The desired poles are set to be identical and are calculated based on the mean of the real and imaginary parts 
      of the original poles, scaled by `poles_gain`.
    - If `plot` is True, a plot of the original and new poles is displayed for visualization.
  """
  # Calculate new poles
  poles = np.linalg.eigvals(A)

  # Set new poles to be more closer to infinity on real axis and close to zero on imaginary axis
  new_x, new_y = -abs(np.mean(np.real(poles))) * poles_gain, abs(np.mean(np.imag(poles))) / poles_gain

  # det(sI - A + LC) = (s - p1)(s - p2)(s - p3)
  s = sp.symbols('s')

  l1, l2, l3 = sp.symbols('l1 l2 l3')
  L = sp.Matrix([[l1], [l2], [l3]])

  # This creates a polynome with the s as symbolic variable 
  # and the coefficients as k1, k2, k3
  left_side_poly = (sp.eye(3) * s - A + L@C).det().simplify()

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
  solution = sp.solve(eqs, (l1, l2, l3))
  L = solution[l1], solution[l2], solution[l3]
  print("\n\033[95mSolution for L:\033[0m")
  print(L, "\n")

  if plot:
    plt.plot(np.real(poles),np.imag(poles),'bx', label='Original poles')
    plt.plot(np.real([p1, p2, p3]),np.imag([l1, l2, l3]),'rx', label='New poles')
    plt.xlabel('Real axis')
    plt.ylabel('Imaginary axis')
    plt.grid(True)
    plt.legend()
    plt.title('Pole Placement')
    plt.show()

  return np.array(L, dtype=float).reshape(-1, 1)