
def __init__(self, M, x0, b, k=2, err_bound=0.01, 
                print_info=True):
    """ Initialize a single LDE problem of dx/dt = Mx + b
    Inputs:
        -  M: matrix of the LDE
        - x0: initial condition vector
        -  b: forcing vector
        -  k: order of Taylor expansion
        - err_bound: error bound for Classiq's coefficient approximation
    """
    self.k = k
    self.err_bound = err_bound
    self.cwd = os.getcwd()
    self.data = None
    
    self.x0_norm, self.x0 = quantum_encode(x0)
    self.b_norm = np.linalg.norm(b)
    self.b = b
    if self.b_norm != 0:
        self.b_norm, self.b  = quantum_encode(b)
    if len(x0) != len(b):
        raise ValueError("x0 and b must be the same length")
    self.M = np.array(M)
    if len(x0) < len(M[0]):
        raise ValueError("M is too big for given x0, b")
    if len(x0) != len(M[0]):
        # extend M to larger hilbert space of x0
        newM = np.identity((len(x0)))
        newM[:len(M[0]),:len(M[0])] = M
        self.M = newM
    
    print(self.x0, self.b)
    print(self.x0_norm, self.b_norm)
    # compute all norms once
    #self.x0_norm = np.linalg.norm(x0)
    #print(self.x0, self.x0_norm)
    
    # use order-2 norm to preserve unitarity
    self.M_norm = np.linalg.norm(M, ord=2)
    A = M / self.M_norm
    self.isUnitary = is_unitary(A)
    if not self.isUnitary:
        raise NotImplementedError("Non-unitary M not yet implemented")
    self.U = [np.linalg.matrix_power(A, i).tolist() for i in range(0,k+1)]
    
    self.nwork = math.ceil(math.log2(len(x0)))
    self.ntaylor = math.ceil(math.log2(self.k+1))
    nqubits = self.nwork + self.ntaylor
    nqubits += 1 if self.b_norm != 0 else 0
    print(f'Needs at least {nqubits} qubits to run!')

    if print_info:
        print(f"Initialized LDE solver with k={k}, err_bound={err_bound}")
        print(f"Initial condition x0={x0} with norm {self.x0_norm}")
        print(f"Matrix M={M} with norm {self.M_norm}")
        print(f"Forcing vector b={b} with norm {self.b_norm}")
        print(f"Matrix A={A} is {'unitary' if self.isUnitary else 'not unitary'}")



def get_Cs(self, M_norm, x0_norm, k, t):
    ''' returns (C_norm, C_amps) when b is zero '''
    C = np.empty(k+1)
    C[0] = x0_norm
    for i in range(1,k+1):
    C[i] = C[i-1] * M_norm * t / i
    # encode amplitudes as probabilities
    Cnorm, C_amps = quantum_vectorize(C)
    return Cnorm, C_amps.tolist()

def unitary_LDE_nob_qmod(self, C_amps, save_qmod=''):
    """ 
    M is unitary AND b = 0 
    """
    @qfunc
    def prepare_registers(work: QArray, taylor: QNum):
        # evolve ancilla into superposition state
        # encode x_0 and Cvals into registers with |0> ancilla
        self.inp_multi_reg_amps(work, self.x0, taylor, C_amps)

    @qfunc
    def do_entangling(work: QArray, taylor: QNum):
        # apply powers of A to taylor register
        for i in range(self.k+1):
            control(taylor == i,
                    lambda: unitary(self.U[i], work))

    @qfunc
    def main(work: Output[QArray], taylor: Output[QNum]):
        # allocate all qubits before doing anything
        allocate(self.nwork, work)
        allocate(self.ntaylor, taylor)
        # apply V^dag U V
        within_apply(lambda: prepare_registers(work, taylor),
                        lambda: do_entangling(work, taylor))
    
    # create the model!
    qmod = create_model(main)
    # store the .qmod code!
    if save_qmod:
        write_qmod(qmod, save_qmod)
    return qmod



def solve(t, save_qmod=''):
    """ Solve the LDE at time t """
    self.M_norm = np.linalg.norm(M, ord=2)
    A = M / self.M_norm
    isUnitary = is_unitary(A)
    if not self.isUnitary:
        raise NotImplementedError("Non-unitary M not yet implemented")
    U = [np.linalg.matrix_power(A, i).tolist() for i in range(0,k+1)]
    
    nwork = math.ceil(math.log2(len(x0)))
    ntaylor = math.ceil(math.log2(self.k+1))
    nqubits = self.nwork + self.ntaylor
    nqubits += 1 if self.b_norm != 0 else 0
    print(f'Needs at least {nqubits} qubits to run!')

    if self.b_norm == 0:
        Cnorm, C_amps = self.get_Cs(self.M_norm, self.x0_norm, self.k, t)
        qmod = self.unitary_LDE_nob_qmod(C_amps, save_qmod)
        return qmod
    else:
        raise NotImplementedError("Non-zero forcing vector b not yet implemented")
        #return self.unitary_LDE_b_qmod(C_amps, save_qmod)
    return None