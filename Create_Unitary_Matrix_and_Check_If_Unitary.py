# Create arbitary M Matrix with dimensions N x N in complex space
def random_matrix(N):
    M = np.random.rand(N, N) + 1.0j * np.random.rand(N, N)
    return M


# Determine matrix if unitary

def unitary_check(M):
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