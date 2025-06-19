import numpy as np

def calculate_N(A, B, C, D, K, x_dot_rows_num, r_rows_num) -> np.ndarray:
  """
  Computes the feedforward gain matrix N for reference tracking in state-space control systems.

  This function constructs an extended system matrix from the provided state-space matrices (A, B, C, D),
  and computes the feedforward gain matrices Nx and Nu. It then combines these with the state feedback gain K
  to obtain the final feedforward gain N, which ensures proper reference tracking.

  Args:
    A (np.ndarray): State matrix of the system.
    B (np.ndarray): Input matrix of the system.
    C (np.ndarray): Output matrix of the system.
    D (np.ndarray): Feedthrough (direct transmission) matrix of the system.
    K (np.ndarray): State feedback gain matrix.
    x_dot_rows_num (int): Number of state variables (rows in the state derivative vector).
    r_rows_num (int): Number of reference inputs (rows in the reference vector).

  Returns:
    np.ndarray: The computed feedforward gain matrix N for reference tracking.
  """
  # Create extended matrix
  AB = np.concatenate((A, B), axis=1)
  CD = np.concatenate((C, D), axis=1)
  extended_matrix = np.concatenate((AB, CD), axis=0)

  extended_x_dot = np.zeros((x_dot_rows_num, r_rows_num), dtype=float)
  extended_y = np.eye((r_rows_num), dtype=float)
  extended_state_matrix = np.concatenate((extended_x_dot, extended_y), axis=0)


  Nx_Nu = np.linalg.inv(extended_matrix) @ extended_state_matrix
  Nx = Nx_Nu[:x_dot_rows_num, :]
  Nu = Nx_Nu[x_dot_rows_num:, :]

  #print(f"Solution for Nx: {Nx}")
  #print(f"Solution for Nu: {Nu}")
  print("\n\033[95mSolution for N:\033[0m")
  print(Nu + K@Nx)

  return Nu + K@Nx