import numpy as np
import math as

def compute_Cm(M, m, x0, t):
    norm_x0 = np.linalg.norm(x0)
    frobenius_norm = np.linalg.norm(M, 'fro')
    Cm = norm_x0 * (frobenius_norm ** m) * (t ** m) / math.factorial(m)
    return Cm


def compute_Dn(M, n, b, t):
    norm_b = np.linalg.norm(b)
    frobenius_norm = np.linalg.norm(M, 'fro')
    Dn = norm_b * (frobenius_norm * t) ** (n - 1) * t / math.factorial(n)
    return Dn


def normalization(M, k, x0, b, t):
    '''
    call this for the kth sum, uses Cm and Dn
    '''
    Cvals, Dvals = [], []
    # compute sum of Cm and Dn
    for m in range(k + 1):
        Cm = compute_Cm(M, m, x0, t)
        Cvals.append(Cm)
        C += Cm ** 2
    for n in range(k + 1):
        Dn = compute_Dn(M, n, b, t)
        Dvals.append(Dn)
        D += Dn ** 2
    C = np.sqrt(C)
    D = np.sqrt(D)
    return C, D, Cvals, Dvals


def V_operator(C, D):
    # N is the normalization factor sqrt(C^2 + D^2)
    N = np.sqrt(C ** 2 + D ** 2)
    V = (1 / N) * np.array([[C, D], [D, -C]])
    return V


# Construct V_S1 and V_S2 matrices
def construct_VS1_VS2(C_vals, D_vals, C, D, k):
    VS1 = np.zeros((k + 1, k + 1), dtype=complex)
    for i in range(k + 1):
        VS1[0, i] = np.sqrt(C_vals[i]) / C

    # Fill the rest of the matrix with arbitrary values to make it unitary
    # We input identity elements
    for i in range(1, k + 1):
        VS1[i, i] = 1.0

    VS2 = np.zeros((k + 1, k + 1), dtype=complex)
    for i in range(1, k + 1):
        VS2[0, i] = np.sqrt(D_vals[i - 1]) / D  # D_vals starts from D1, so shift index
    VS2[0, k] = 0  # The last element is 0

    # Fill the rest of the matrix with arbitrary values to make it unitary
    for i in range(1, k + 1):
        VS2[i, i] = 1.0

    return VS1, VS2


def construct_W_WS1_WS2(V, VS1, VS2):
    '''
    Apply conjugate transpose to each of the V operators
    '''
    W = V.conj().T
    WS1 = VS1.conj().T
    WS2 = VS2.conj().T
    return W, WS1, WS2