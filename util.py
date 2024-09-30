import numpy as np
import math
# for plot_op
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mplcm
import matplotlib.colorbar as mplcb
SAVEDIR = "."
SAVEEXT = ".png"
from classiq.synthesis import synthesize, set_constraints, set_execution_preferences, set_preferences, show
from classiq.executor import execute
from classiq.execution import ExecutionPreferences, IBMBackendPreferences, IBMBackendProvider
from classiq.interface.generator.quantum_program import QuantumProgram
from classiq import Constraints, Preferences
import time

#################################
#     UTILITY FUNCTIONS         #
#################################

#       general quantum         #
def is_unitary(M: np.ndarray) -> bool:
    ''' Returns whether M is unitary '''
    return np.allclose(np.identity(M.shape[0]), M @ M.conj().T)

def quantum_vectorize(V: list | np.ndarray) -> tuple[float, np.ndarray]:
    """ Convert vector of floats to (real) amplitude quantum vector """
    norm = np.sqrt(np.sum(V))
    # encode as amplitudes in quantum vector
    A = np.sqrt(V) / norm
    return norm, A

def quantum_unvectorize(norm, A: list | np.ndarray) -> np.ndarray:
    """ Convert quantum vector to real vector 
    note: cannot recover any phase information! """
    return (norm * A)**2

def quantum_encode(x: list | np.ndarray) -> tuple[float, list[float]]:
    """
    Create a valid normalized quantum state from an input list of complex numbers.
     - list is assumed to be probabilities/ratios of the state and 
        will be converted to quantum amplitudes
    ex. [1,2,3] treated as probability vector [1,2,3]/6
    """
    n = len(x)
    # round up to the nearest power of 2
    m = 2 ** math.ceil(math.log2(n))
    # pad with zeros
    x = list(x) + [0] * (m - n)
    x = np.array(x)
    # replace nonsensical zero-state with mixed state
    if np.sum(x) == 0:
        x = np.ones(len(x))
    # normalize
    # [1, 2, 3] -> [1, sqrt(2), sqrt(3)]/sqrt(6)
    norm, x = quantum_vectorize(x)
    assert isinstance(x, np.ndarray)  # for PyLance stupid type-setting
    # convert to Classiq-compatible list
    return norm, x.tolist()

#   parsing bitstrings   #
def all_bitstrings(N, lsb_right=True):
    """ return strings of all possible N-bit states
     N=0 returns []
     N=2 returns ['00','01','10','11'] for lsb_right
             and ['00','10','01','11'] for lsb_left
    """
    ret = []
    if N <= 0: 
        raise ValueError(f'number of qubits must be >= 1 but you used {N}')
    for i in range(2**N):
        s = str(bin(i))[2:]  #  first 2 chars of binary string repr are '0b'
        # left-fill s with 0s
        s = '0'*(N-len(s)) + s
        if not lsb_right: 
            s = s[::-1]
        ret.append(s)
    return ret

def get_work_strings(nqubits, work_idxs, lsb_right=True, ancilla_bits=None):
    """ 
    generate all possible 'work' qubit states out of nqubits-long bitstring
    with the rest
    out of a full state vector nqubits long, generate all possible work qubit states
    which have qubits at indices place.
     - work_idxs = locations of work qubits within nqubit bitstring
     - ancilla_bits: optionally specify len(nqubits)-len(work_idxs) string
    
    ex: nqubits = 6, work_idxs = (2,3)
        lsb_right=True has Q-indices  543210 --> [000000,000100,001000,001100]
        lsb_right=False has Q-indices 012345 --> [000000,001000,000100,001100]
    """
    w = len(work_idxs)  # nqubits of work part of string
    a = nqubits - w   # nqubits of ancilla part of string
    ret = []
    # ancillas are usually all zero if not specified
    if ancilla_bits is None:
        ancilla_bits = '0'*a
    # assume ancilla bits are ALWAYS SPECIFIED in default lsb_right notation
    elif not lsb_right:
        # so we must flip the bit order for lsb_left
        ancilla_bits = ancilla_bits[::-1]
    if len(ancilla_bits) != a:
        raise ValueError(f'provided ancilla-string is {len(ancilla_bits)}-bit but expected an ({nqubits}-{w}={a})-bit string')

    # sort in ascending order for lsb_left (1,2,3)
    # and descending order for lsb_right (3,2,1)
    work_idxs = sorted(work_idxs, reverse=lsb_right)
    for work_string in all_bitstrings(w, lsb_right):
        # ancillas are desired to be all zero...
        string = ''
        i = 0  # position in work_string
        j = 0  # position in ancilla_string
        for q in range(nqubits):
            if lsb_right:  # idxs = 543210
                qubit_idx = nqubits - q - 1
            else:          # idxs = 012345
                qubit_idx = q
            # done w work qubit, add ancilla only
            if i == w:
                string += ancilla_bits[j]
                j += 1
                continue
            # done w ancilla, add work only
            if j == a:
                string += work_string[work_idxs[i]]
                i += 1
                continue
            # at next work qubit
            if qubit_idx == work_idxs[i]:
                string += work_string[i]
                i += 1
            # at next ancilla qubit
            else:
                string += ancilla_bits[j]
                j += 1
        ret.append(string)
        # make full string
    return ret


#       classic-specific         #
def write_qprog(qprog, fname):
    # classiq doesn't have a built-in way to write a qprog to a file for some reason
    file = open(fname+'.qprog',"w")
    file.write(qprog)
    file.close()

def classiq_to_vec(N, rdict, register_name='work', print_info=True):
    """
    returns the amplitude vector from the classiq results
    - N: normalization factor to reconstruct vector with
    - rdict: result dictionary from classiq

    assume that desired vector V = [a b] is encoded 
    in quantum state Q=V/(N**2), where Q is a 
    subset of the total state of the quantum system S
    
    we count probability P as
    N^2 is total nornmaliz
    P=(a^2/N^4 b^2/N^4)
    so we can recover V as
    V = sqrt(P) * N^2

    N^2 = total norm of Q = [a b ,,,,]
    N^2 = a^2 + b^2 + ...
    """
    total_counts = rdict.num_shots
    # reconstruct / parse state svector
    out_map = rdict.output_qubits_map   
    vec_indices = out_map[register_name]     # tuple of indices
    nbits = 2**len(vec_indices)
    y = np.zeros(nbits)
    lsb_right = rdict.counts_lsb_right  # whether output map starts from right or left of string
    counts_dict = rdict.counts
    # set all other qubits to be 0, then reconstruct state vector from counts
    nqubits = len(list(counts_dict.keys())[0])
    # only care about all ancillas being zero
    work_strings = get_work_strings(nqubits, vec_indices, lsb_right)
    # iterate from least-significant to most-significant bitstring
    for j, work_string in enumerate(work_strings):
        if print_info:
            print(f'parsing |{work_string}> as x[{j}]')
        if work_string not in counts_dict:
            print(f"Work string {work_string} not found in counts!")
            continue
        # convert to probabilities = amp^2
        y[j] = counts_dict[work_string] / total_counts
    # convert to amplitudes and multiply by total normalization to get to vector yt
    return np.sqrt(y) * N**2

def gen_fake_classiq_dict(P, nqubits=4, nshots=10000, register_name='work'):
    """P: probability vector
    assumes lsb_right 
    
    P: probability vector with 2^(nwork) entries
    nqubits: total # of qubits in system
             nqubits - log2(len(P)) = # ancilla qubits
    """
    nwork = np.log2(len(P))
    output_qubits = {register_name: list(range(nwork)),
                     'ancilla': list(range(nwork,nqubits))}
    all_strings = all_bitstrings(nqubits)
    counts_dict = {}
    ret = {'num_shots': nshots, 
           'output_qubits_map': output_qubits,
           'counts_lsb_right': True,
           'counts': counts_dict}

    # turn dictionary into class
    return DictObject(ret)


def run_qmod(qmod, opt='depth', nshots=10000, job_name='',
             save_qprog='', sim='Classiq', open_circuit=False, print_info=True):
    """ optimize qmod with constraints and run it on a simulator 
        - sim = 'Classiq' or 'IBM', both classical simulators though
    """
    if sim == 'Classiq':
        # set constraints and preferences
        max_width = 25#; opt = 'depth'; nshots = 10000; job_name = 'LDE_U_no-b_t=0.143'
        qmod = set_preferences(qmod,
            Preferences(backend_service_provider="Classiq", backend_name="simulator"))
    elif sim == 'IBM':
        # REPLACE WITH YOUR OWN IBM API TOKEN!!
        token = '0920786a03ff0a185aca85f5c4a4be13232efe6641aa5685381a1c3984ec1ed1a7fdd85fbf1967af41bebbf30f321eadb4fb7188eb830a91100133a65e9a9fb1'
        max_width = 127
        qmod = set_preferences(qmod,
            IBMBackendPreferences(
                backend_name="ibm_strasbourg",
                access_token=token,
                provider=IBMBackendProvider(hub = "ibm-q", group = "open", project = "main"))
        )
    qmod = set_constraints(qmod,
        Constraints(max_width=max_width, optimization_parameter=opt))
    qmod = set_execution_preferences(qmod,
        ExecutionPreferences(num_shots=nshots, job_name=job_name, 
                            random_seed='767'))
    qmod = set_preferences(qmod, Preferences(timeout_seconds=600, optimization_timeout_seconds=120))
    qprog = synthesize(qmod)
    # synthesize the circuit
    start_time = time.time()
    if print_info:
        print(f"Running {job_name}")
    qprog = synthesize(qmod)
    QP = QuantumProgram.from_qprog(qprog)
    circuit_width = QP.data.width
    circuit_depth = QP.transpiled_circuit.depth
    end_time = time.time()

    if print_info:
        print(f"\tcircuit synthesized in {end_time-start_time:.2f}s: width={circuit_width},depth={circuit_depth}")
    # open in viewer
    if open_circuit: 
        show(qprog)
    # save generated quantum program
    if save_qprog:
        write_qprog(qprog, save_qprog)
    start_time = time.time()
    
    job = execute(qprog)
    if print_info:
        print(
            f"\tjob with {job.num_shots} shots is {job.status} on provider-backend={job.provider}-{job.backend_name} \n\tand can be accessed at {job.ide_url}"
        )
    results = job.result()[0].value
    end_time = time.time()
    if print_info:
        print(f"\tran in {end_time-start_time:.2f}s")
    return results

#
# PREVIOUS UTILITY FUNCTIONS
# BY BEN KROUL
#
def uFormat(number, uncertainty=0, figs = 4, shift = 0, FormatDecimals = False):
    """
    Returns "num_rounded(with_sgnfcnt_dgts_ofuncrtnty)", formatted to 10^shift
    According to section 5.3 of "https://pdg.lbl.gov/2011/reviews/rpp2011-rev-rpp-intro.pdf"

    Arguments:
    - float number:      the value
    - float uncertainty: the absolute uncertainty (stddev) in the value
       - if zero, will format number to optional number of sig_figs (see figs)
    - int shift:  optionally, shift the resultant number to a higher/lower digit expression
       - i.e. if number is in Hz and you want a string in GHz, specify shift = 9
               likewise for going from MHz to Hz, specify shift = -6
    - int figs: when uncertainty = 0, format number to degree of sig figs instead
       - if zero, will simply return number as string
    - bool FormatDecimals:  for a number 0.00X < 1e-2, option to express in "X.XXe-D" format
             for conciseness. doesnt work in math mode because '-' is taken as minus sign
    """
    num = str(number); err = str(uncertainty)
    
    sigFigsMode = not uncertainty    # UNCERTAINTY ZERO: IN SIG FIGS MODE
    if sigFigsMode and not figs: # nothing to format
        return num
    
    negative = False  # add back negative later
    if num[0] == '-':
        num = num[1:]
        negative = True
    if err[0] == '-':
        err = err[1:]
    
    # ni = NUM DIGITS to the RIGHT of DECIMAL
    # 0.00001234=1.234e-4 has ni = 8, 4 digs after decimal and 4 sig figs
    # 1234 w/ ni=5 corresponds to 0.01234
    ni = ei = 0  
    if 'e' in num:
        ff = num.split('e')
        num = ff[0]
        ni = -int(ff[1])
    if 'e' in err:
        ff = err.split('e')
        err = ff[0]
        ei = -int(ff[1])

    if not num[0].isdigit():
        print(f"uFormat: {num} isn't a number")
        return num
    if not err[0].isdigit():
        err = '?'

    # comb through error, get three most significant figs
    foundSig = False; decimal = False
    topThree = ""; numFound = 0
    jErr = ""
    for ch in err:
        if decimal:
            ei += 1
        if not foundSig and ch == '0': # dont care ab leading zeroes
            continue  
        if ch == '.':
            decimal = True
            continue
        jErr += ch
        if numFound >= 3:  # get place only to three sigfigs
            ei -= 1
            continue
        foundSig = True
        topThree += ch
        numFound += 1
    
    foundSig = False; decimal = False
    jNum = ""
    for ch in num:
        if decimal:
            ni += 1
        if not foundSig and ch == '0': # dont care ab leading zeroes
            continue
        if ch == '.':
            decimal = True
            continue
        jNum += ch
        foundSig = True
    if len(jNum) == 0:  # our number is literally zero!
        return '0'
    
    # round error correctly according to PDG
    if len(topThree) == 3:
        nTop = int(topThree)
        if nTop < 355: # 123 -> (12.)
            Err = int(topThree[:2])
            if int(topThree[2]) >= 5:
                Err += 1
            ei -= 1
        elif nTop > 949: # 950 -> (10..)
            Err = 10
            ei -= 2
        else:  # 355 -> (4..)
            Err = int(topThree[0])
            if int(topThree[1]) >= 5:
                Err += 1
            ei -= 2
        Err = str(Err)
    else:
        Err = topThree

    n = len(jNum); m = len(Err)
    nBefore = ni - n  #; print(num, jNum, n, ni, nBefore)
    eBefore = ei - m  #; print(err, Err, m, ei, eBefore)
    if nBefore > eBefore:  # uncertainty is a magnitude larger than number, still format number
        if not sigFigsMode:
            print(f'Uncrtnty: {uncertainty} IS MAGNITUDE(S) > THAN Numba: {number}')
        Err = '?'
    if sigFigsMode or nBefore > eBefore:
        ei = nBefore + figs

    # round number to error
    d = ni - ei 
    if ni == ei: 
        Num = jNum[:n-d]
    elif d > 0:  # error has smaller digits than number = round number
        Num = int(jNum[:n-d])
        if int(jNum[n-d]) >= 5:
            Num += 1
        Num = str(Num)
    else:  # error << num
        Num = jNum
        if ei < m + ni:
            Err = Err[n+d-1]
        else:
            Err = '0'
    if ni >= ei: ni = ei  # indicate number has been rounded
    
    n = len(Num)
    # if were at <= e-3 == 0.009, save formatting space by removing decimal zeroes
    extraDigs = 0
    if not shift and FormatDecimals and (ni-n) >= 2:
        shift -= ni - n + 1
        extraDigs = ni - n + 1
    
    # shift digits up/down by round argument
    ni += shift
    end = ''

    # there are digits to the right of decimal and we dont 
    # care about exact sig figs (to not format floats to 0.02000)
    if ni > 0 and sigFigsMode:
        while Num[-1] == '0':
            if len(Num) == 1: break
            Num = Num[:-1]
            ni -= 1
            n -= 1
    
    if ni >= n:   # place decimal before any digits
        Num = '0.' + "0"*(ni-n) + Num
    elif ni > 0:  # place decimal in-between digits
        Num = Num[:n-ni] + '.' + Num[n-ni:]
    elif ni < 0:  # add non-significant zeroes after number
        end = 'e'+str(-ni)
    if extraDigs:  # format removed decimal zeroes
        end = 'e'+str(-extraDigs)
    
    if negative: Num = '-' + Num  # add back negative
    if not sigFigsMode:
        end = '(' + Err + ')' + end
    return Num + end


def plot_op(operators: list, titles=[], saveplot=False, cmap_name="turbo", box_spec=False):
    """ Plot 2-dimensional operators
    Inputs: 
    - operators: takes in ndarray or list of ndarrays 
    - titles: title or list of titles for multiple ops 
    - saveplot: True or string to name file of plot
    - cmap_name: name of mpl.colormap to use
    """
    if not isinstance(operators, list):
        operators = [operators]
    if not isinstance(titles, list):
        titles = [titles]
    nops = len(operators)
    fig = plt.figure(figsize=(8,4*nops))
    gs = fig.add_gridspec(nops,3,width_ratios=[20,20,1])
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.1)
    # get abs max of all operators
    absmax = 0
    for op in operators:
        themin = min(op.real.min(), op.imag.min())
        themax = max(op.real.max(), op.imag.max())
        amax = max(abs(themin),abs(themax))
        absmax = max(absmax, amax)
    # make colorbar
    cax = fig.add_subplot(gs[:,2])
    norm = mcolors.Normalize(vmin=-absmax, vmax=absmax)
    cm_map = mplcm.ScalarMappable(norm=norm, cmap=cmap_name)
    cb = mplcb.ColorbarBase(cax, cmap=cmap_name, norm=norm, orientation='vertical')
    # plot operators
    for i in range(nops):
        operator = operators[i]
        ax1 = fig.add_subplot(gs[i,0])
        ax2 = fig.add_subplot(gs[i,1])
        ax1.matshow(operator.real, cmap=cmap_name, vmin = -absmax, vmax=absmax)
        ax2.matshow(operator.imag, cmap=cmap_name, vmin = -absmax, vmax=absmax)
        title = titles[i] if len(titles) > i else "_DEF_"
        ax1.set(xticks=[],yticks=[])
        ax1.set_title("Re{ "+title+" }")
        ax2.set(xticks=[],yticks=[])
        ax2.set_title("Im{ "+title+" }")
        if box_spec:
            for (i, j), val in np.ndenumerate(operator.real):
                ax1.text(j, i, uFormat(val,0), ha='center', va='center', color=('black' if abs(val) < absmax/2 else 'white'))
            for (i, j), val in np.ndenumerate(operator.imag):
                ax2.text(j, i, uFormat(val,0), ha='center', va='center', color=('black' if abs(val) < absmax/2 else 'white'))
    if saveplot:
        if isinstance(saveplot, str):
            plt_name = SAVEDIR + "/"+saveplot.replace(" ","_").replace("$","")+"_2d"+SAVEEXT
        else:
            plt_name = SAVEDIR + "/"+title.replace(" ","_").replace("$","")+"_2d"+SAVEEXT
        plt.savefig(plt_name,bbox_inches="tight")
        print(f'saved figure {plt_name}')
    plt.show()