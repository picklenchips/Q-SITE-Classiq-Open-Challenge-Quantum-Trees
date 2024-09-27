import numpy as np
import math


#
# UTILITY FUNCTIONS
#

# from Jacob's code
def is_unitary(M: np.ndarray) -> bool:
    ''' Returns whether M is unitary '''
    return np.allclose(np.identity(M.shape[0]), M @ M.conj().T)

def quantum_encode(x: list | np.ndarray) -> np.ndarray:
    """
    Create a valid normalized quantum state from an input list of complex numbers.
     - list is assumed to be probabilities/ratios of the state and not amplitudes
    """
    n = len(x)
    # round up to the nearest power of 2
    m = 2 ** math.ceil(math.log2(n))
    # pad with zeros
    x = list(x) + [0] * (m - n)
    x = np.array(x)
    # replace nonsensical all zeros with mixed state
    if np.linalg.norm(x) == 0:
        x = np.ones(len(x))
    # normalize and convert to Classiq-compatible list
    x = x / np.linalg.norm(x)
    return x.tolist()