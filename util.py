import numpy as np

# from Jacob's code
def unitary_check(M: np.ndarray) -> bool:
    '''
    Input: Random matrix M
    Output: Boolean variable: True or False
    '''
    # Conjugate transpose
    M_dag = M.conj().T

    # Multiply M and M_dag
    MM_dag = np.dot(M, M_dag)

    # Create identity matrix
    Identity = np.eye(M.shape[0])

    return np.allclose(Identity, MM_dag)