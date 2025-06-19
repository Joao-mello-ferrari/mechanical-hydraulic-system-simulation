import numpy as np
from numpy.linalg import matrix_rank

def check_controllability(A, B):
    """
    Check if the system is controllable.
    
    Parameters:
    A (np.ndarray): State matrix.
    B (np.ndarray): Input matrix.
    
    Returns:
    bool: True if the system is controllable, False otherwise.
    """
    n = A.shape[0]
    controllability_matrix = np.hstack([B, A @ B, A @ A @ B, A @ A @ A @ B])
    rank_C = matrix_rank(controllability_matrix)
    return rank_C == n
def check_observability(A, C):
    """
    Check if the system is observable.
    
    Parameters:
    A (np.ndarray): State matrix.
    C (np.ndarray): Output matrix.
    
    Returns:
    bool: True if the system is observable, False otherwise.
    """
    n = A.shape[0]
    observability_matrix = np.vstack([C, C @ A, C @ A @ A, C @ A @ A @ A])
    rank_O = matrix_rank(observability_matrix)
    return rank_O == n