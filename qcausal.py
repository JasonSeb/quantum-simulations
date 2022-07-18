""" 
# Author  : Jason Connie
# Updated : July 2022

Library to help simulate various quantum computing operations.

Methods cover operations for basic linear algebra, the state vector formalism 
of quantum computing, the density matrix formalism, as well as the quantum 
causal formalism that makes use of the Choi-Jamiolkowski isomorphism.

"""

import numpy as np
import random as rd

###############################################################################
########## Section defining basic methods for general linear algebra ##########
###############################################################################

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = (1/np.sqrt(2))*np.array([[1, 1], [1, -1]], dtype=complex)


""" Method to give the kronecker product of all inputted matrices """
def tensor(*args):
    W = np.array([1], dtype=complex)

    for i in args:
        W = np.kron(W, i)

    return W




""" Method to return the tensor exponential of a matrix
    For example, I tensored to the power of 2 is tensor(I,I) """
def tensor_exp(W, power):
    W2 = W
    
    for i in range(power-1):
        W2 = tensor(W2,W)
    
    if (power==0):
        W2=1
    
    return W2




""" Method to count the number of zero entries within a given matrix """
def zero_count(mat):
    count = 0
    mat_dim = np.shape(mat)

    for i in range(mat_dim[0]):
        for j in range(mat_dim[1]):
            if (abs(mat[i, j]) < 0.00000000000000001):
                count += 1
    return count




""" Method to check the purity of a state
    A value pf 1 is maximally pure
    A value of 1/d is maximally impure, where d is the dimension of the density matrix rho
    ie: 1/d <= purity <= 1 is our range, with higher values indicate greater purity
    """
def purity_check(rho):
    purity = np.trace(rho@rho)
    
    # purity should only be real, and neglible imaginary components are treated as float errors
    if (np.abs(np.imag(purity))<0.000005): 
        purity = np.real(purity)
    
    return purity




""" Method to return the complete trace of a matrix
    The only benefit over using the numpy version is that potential imaginary floating errors are removed """
def trace(rho):
    tr = np.trace(rho)
    
    # We once more round any neglible imaginary components, presuming them to be float errors
    if (np.imag(tr)<0.000005):
        return round(np.real(tr),16)
    else:
        return tr




""" Method to take the partial trace of a state, represented by a density matrix
    The index of the subsystem we wish to trace out must be specified, indexing from 0 """
def ptrace(rho, subsystem_num):
    num_qubits = int(np.log2(np.shape(rho)[0]))        # Number of qubits in total system

    # Return None if someone tries to trace out an invalid system
    if (num_qubits<=subsystem_num or subsystem_num<0):
        print("Issue with partial tracing indexing")
        return None

    # We trace out the indicated system
    b0 = np.array([[1],[0]])    # basis vector 0
    b1 = np.array([[0],[1]])    # basis vector 1
    
    m0 = np.array([1])
    m1 = np.array([1])        
    
    for i in range(num_qubits):
        if (i==subsystem_num):
            m0 = tensor(m0,b0)
            m1 = tensor(m1,b1)
        else:                   # We place an identity in every subsystem we aren't tracing out
            m0 = tensor(m0,I)
            m1 = tensor(m1,I)
            
    rho = (m0.T)@rho@m0 + (m1.T)@rho@m1  # Remember that @ gives matrix multiplication!
    return rho


    

""" Method to take the partial trace of a matrix, given a range of subsystems to trace out """
def ptrace_list(rho, subsystem_list):
    subsystem_list = np.sort(subsystem_list)
    subsystem_num  = subsystem_list[-1]
    subsystem_list = subsystem_list[:-1]
    
    rho = ptrace(rho, subsystem_num)
    
    if (len(subsystem_list)==0):
        return rho
    else:
        return ptrace_list(rho, subsystem_list)
    



""" Method to trace out the ith of 4 input/output spaces, where i is represented by the variable 'iospace_num'
    The prefix 'c' indicates customised, as this is intended primarily for biparite process matrices
    The dimensions of each input and output space default to 2, but alternate values can be inputted """
def c_ptrace(rho, iospace_num, iospace_dims=[2,2,2,2]):
    iospace_qubits = np.log2(iospace_dims).astype(int) # The number of qubits for each input and output space
    
    start = sum(iospace_qubits[:iospace_num])
    end   = sum(iospace_qubits[:(iospace_num+1)])

    subsystem_list = list(range(start,end))
    
    return ptrace_list(rho, subsystem_list)




""" Method to trace out multiple input and/or output spaces at once, for bipartite process matrices 
    The 'c' prefix once more indicates customised """
def c_ptrace_list(rho, iospace_list, iospace_dims=[2,2,2,2]):
    iospace_list = np.sort(iospace_list)
    iospace_num  = iospace_list[-1]
    iospace_list = iospace_list[:-1]
    
    rho = c_ptrace(rho, iospace_num, iospace_dims)
    
    if (len(iospace_list)==0):
        return rho
    else:
        return c_ptrace_list(rho, iospace_list, iospace_dims)




""" Method to check if an inputted set of n vectors forms a basis of n orthogonal vectors
    (where n is the number of basis vectors needed for the Hilbert space considered)
"""
def basis_check(basis):
    # We make sure there are no zero vectors
    basis = [b for b in basis if not all(b==0)]

    
    # We see if the number of vectors given are enough to form a basis. If there are too few or too many, we return False
    n = len(basis[0])
    numVectors = len(basis)
    
    if (numVectors != n):
        return False

    
    # We see if all the basis vectors are orthogonal. The final total will be zero if all vectors are orthogonal
    total = 0
    
    for i in range(n):
        basis1 = np.conj(basis[i]).T
        for j in range(i+1,n):
            basis2 = basis[j]
            innerProduct = abs( (basis1@basis2)[0,0] )
            total += innerProduct
    
    
    # If the total is zero (up to some flaoting point error), we know the input is indeed a basis and return True
    if (total < 0.000000000000000001):
        return True
    else:
        return False


    

###############################################################################
######### Section defining basic methods for handling quantum states ##########
###############################################################################

""" Method that transforms a state vector into a density matrix """
def state2dm(ket_state):
    bra_state = np.conj(ket_state.T)
    return ket_state*bra_state




""" Method to produce a random separable state, of n qubits """
def separable_state(num_qubits):
    # We create a state vector of the form tensor(a1,a2,a3*,...,an) where n=2**num_qubits
    # and each ai is a random single qubit state. However, as each single qubit only needs 
    # 2 real parameters to reproduce (recall the Bloch sphere), we need only 2n parameters overall
    state = np.array([1])
    for i in range(num_qubits):
        psi = rd.uniform(0,np.pi+0.00001)
        phi = rd.uniform(0,2*np.pi)
        temp_state = np.array([[np.cos(psi/2)],
                               [np.exp(1j*phi)*np.sin(psi/2)]])
        state = np.kron(state, temp_state)
    
    return state




""" Method that 'checks' whether a state is entangled
    If it returns True, the state is definitely entangled 
    If it returns False, the state could be either entangled or separable 
    
    The purity measure acts as a measure of entanglement one will be satisfied with, 
    with greater impurity implying greater entanglement """
def entanglement_check(state, purity_measure):
    num_qubits = int(np.log2(len(state)))
    
    if (num_qubits>1):        # A state can only be entangled if it is more than 1 qubit in size
        rho = state2dm(state)
        traced_rho = ptrace(rho,1)
        purity = purity_check(traced_rho)
        # Suggestion: go with purity_measure=0.65 for 2-4 qubits, 0.75 for 5-7, and 0.85 for 8-9
        if (purity < purity_measure):
            return True
        else:
            return False
    
    else:     # If a single qubit state is given, or otherwise, we return False
        print("Issue with entanglement checker use")
        return False




""" Method to produce a random entangled state
    The purity measure acts as a threshold for how entangled we want the state to be at minimum, 
    as it indicates how impure we wish the state to be after tracing out the first qubit """
def entangled_state(num_qubits, purity_measure=0.75, iters=15):
    
    # Create the real and imaginary parts of a state vector, combine and normalize them
    vec_real = np.random.rand(int(2**num_qubits),1)
    vec_imag = np.random.rand(int(2**num_qubits),1)
    state = vec_real + 1j*vec_imag
    state = (1/np.linalg.norm(state))*state
    
    # Return the state if it is sufficiently entangled
    if entanglement_check(state, purity_measure):
        return state
        
    # This process is not optimized, and can be costly. We stop after a certain number of iterations
    else:
        iters -= 1
        if (iters>0):
            return entangled_state(num_qubits)
        else:
            print("Sufficient entanglment isn't being made, perhaps raise the purity")
            return None
        



""" Method to normalize a matrix """
def normalize(rho):
    trc = np.abs(np.trace(rho))
    return (1/trc)*rho




""" Method to return the probabilities of all possible measurement outcomes of measuring a given state
    If a basis is not given, the computational basis is assumed
    
    The state to be measured can be inputted as a density matrix or state vector
"""
def measure(state, basis=None):
    num_qubits = int(np.log2(len(state)))    

    # If no basis is given, we use the computational basis of the appropriate dimensions 
    if (basis==None):        
        b0 = np.array([[1],[0]])    # basis vector |0)
        b1 = np.array([[0],[1]])    # basis vector |1)
        basis = tensored_basis_set([b0,b1], num_qubits)
    
    # If a basis is given but is incorrect, we print an error message and return None
    elif (basis_check(basis) == False):
        print("This is an incomplete, overcomplete or invalid basis")
        return None
    
    # Given a basis of n vectors, we now calculate the probability of each of the n outcomes
    measurement_results = [None]*len(basis)
    
    # If the inputted 'state' is a state vector rather, we set it to a density matrix and assign it to 'rho' 
    # If the inputted 'state' is already a density matrix, we assign it to 'rho' with no adjustment needed
    if (1 in np.shape(state)):
        rho = state2dm(state)
    else:
        rho = state
    
    # Loop to work out the measurement probabilities
    for i in range(len(basis)):
        Mi         = basis[i]@np.conj(basis[i].T)   # The measurement operator for the ith measurement
        ith_result = trace(Mi@rho)
        
        measurement_results[i] = np.real(ith_result)
    
    return measurement_results
    
    
    
    
###############################################################################
### Section defining basic methods for quantum gates and circuit generation ###
###############################################################################

""" Method to return a single qubit gate, given a list of four real parameters.
    Equivalent to U = e^(i*alpha)*Rz(beta)Ry(gamma)Rz(delta), a universal reresentation for single qubit gates.
    Details can be seen on page 176 of Nielsen and Chuang
"""
def single_qubit_gate(gate_params):
    alpha = gate_params[0]
    beta  = gate_params[1]
    delta = gate_params[2]
    gamma = gate_params[3]
    
    a =  np.exp(complex(0, alpha - beta/2 - delta/2))*np.cos(gamma/2)
    b = -np.exp(complex(0, alpha - beta/2 + delta/2))*np.sin(gamma/2)
    c =  np.exp(complex(0, alpha + beta/2 - delta/2))*np.sin(gamma/2)
    d =  np.exp(complex(0, alpha + beta/2 + delta/2))*np.cos(gamma/2)
    
    gate = np.array([[a,b],[c,d]])
    
    return np.round(gate,12)




""" Method to return a unitary gate of n qubits size: a single qubit gate is applied to the specified qubit, 
    and identity matrices are applied to everything else.
    
    The single qubit gate can be given either as a matrix or by its parameters ('gate_or_params')
    
    'target_qubit' indicates the index of the qubit our gate acts upon, where we index from 0 as usual """
def unitary(gate_or_params, target_qubit, total_num_qubits):
    
    # If we are given parameters, we generate the appropriate gate
    if len(np.shape(gate_or_params))==1:
        U = single_qubit_gate(gate_or_params)
    else:
        U = gate_or_params
    
    u_total = np.array([1])
    
    for i in range(total_num_qubits):
        if (i==target_qubit):
            u_total = np.kron(u_total,U)
        else:
            u_total = np.kron(u_total,I)
    
    return u_total




""" Method to return the controlled application of a single qubit gate
    Either the single qubit gate or its parameters can be inputted (as 'gate_or_params')
    The final gate will have the dimension 2**totalNumQubits 
    
    Which qubit acts as the control, and which acts as the target, must be specified """
def controlled_unitary(gate_or_params, target_qubit, control_qubit, total_num_qubits):
    
    if len(np.shape(gate_or_params))==1:
        U = single_qubit_gate(gate_or_params)    # the desired unitary gate
    else:
        U = gate_or_params
        
        
    m0 = np.array([[1,0],[0,0]])
    m1 = np.array([[0,0],[0,1]])
        
    cu1 = 1
    cu2 = 1

    for i in range(total_num_qubits):
        if (i==control_qubit):
            cu1 = np.kron(cu1,m0)
        else:
            cu1 = np.kron(cu1,I)

    for j in range(total_num_qubits):
        if (j==target_qubit):
            cu2 = np.kron(cu2,U)
        elif (j==control_qubit):
            cu2 = np.kron(cu2,m1)
        else:
            cu2 = np.kron(cu2,I)

    return np.array(cu1+cu2)




""" Method to return parameters for a circuit of only Hadamard and controlled NOT gates
    (To be used as a starting point for training variational circuits)

    circ_params[i] returns the the parameters of the ith block 
    circ_params[i][0] returns the single gates for that block
    circ_params[i][1] returns the controlled gates for that block """
def seed_circuit(numQubits, numBlocks):
    
    circ_params = [None]*numBlocks
    
    for i in range(numBlocks):
        gates  = [[-np.pi/2, 0    , -np.pi, np.pi/2]]*numQubits    # Hadamard gates
        cgates = [[-np.pi/2, np.pi, 0     , np.pi  ]]*numQubits    # controlled X gates (NOT gates)
        block = [gates,cgates]
        circ_params[i] = block
    return circ_params




""" Method that returns the parameters for a random circuit
    (Also to be used as the starting point for training a variational circuit)

    circ_params[i] returns the the parameters of the ith block 
    circ_params[i][0] returns the single gates for that block
    circ_params[i][1] returns the controlled gates for that block """
def rand_circuit(numQubits,numBlocks):
    circ = [None]*numBlocks
    for i in range(numBlocks):
        gates  = [None]*numQubits
        cgates = [None]*numQubits
        
        for j in range(numQubits):
            alpha = 0 # The alpha parameters of non-controlled gates simply affect the global phase, and can be ignored 
            beta  = rd.uniform(-np.pi,np.pi)
            delta = rd.uniform(-np.pi,np.pi)
            gamma = rd.uniform(-np.pi,np.pi)
            gates[j]  = [alpha,beta,delta,gamma]
            
            calpha = rd.uniform(-np.pi,np.pi)   # alpha, beta, delta and gamma for controlled gate
            cbeta  = rd.uniform(-np.pi,np.pi)
            cdelta = rd.uniform(-np.pi,np.pi)
            cgamma = rd.uniform(-np.pi,np.pi)            
            cgates[j] = [calpha,cbeta,cdelta,cgamma] 
        
        block = [gates,cgates]
        circ[i] = block
    return circ




""" Method to turn the parameters of a circuit object into its matrix representation """
def circuit_to_matrix(circ):
    numBlocks = len(circ)
    numQubits = len(circ[0][0])
    
    circ_matrix  = np.eye(2**numQubits)
    
    for i in range(numBlocks):
        #cgates = circ[i][1]
        gates_product  = np.array([1]) # ith block gate matrix
        cgates_product = np.eye(2**numQubits)
        for j in range(numQubits):
            control = j
            target  = (j+1)%numQubits
            
            U    = single_qubit_gate(circ[i][0][j])
            conU = controlled_unitary(circ[i][1][j], target, control, numQubits)
            
            gates_product  = tensor(gates_product, U)
            cgates_product = cgates_product@conU
    
        block = cgates_product@gates_product
        circ_matrix = block@circ_matrix            
    
    return circ_matrix        




###############################################################################
######## Section for specific gates and the Quantum Fourier Transform #########
###############################################################################

""" Method to return the parameters of commonly named single qubit gates 
    - the Hadamard gate 'H'
    - the identity matrix 'I'
    - the Pauli matrices 'X', 'Y', 'Z'
    - the phase gate 'S'
    - the pi/8 gate 'T'
    
    NOTE: Different parameters can give rise to the same gate, so these values are not unique
"""
def named_gate_parameters(gate_name):
    gate_name = gate_name.upper()
    
    if gate_name == 'H':
        return [-np.pi/2, 0, -np.pi, np.pi/2]
    elif gate_name == 'I':
        return [0, 0, 0, 0]
    elif gate_name == 'X':
        return [-np.pi/2, np.pi, 0, np.pi]
    elif gate_name == 'Y':
        return [-np.pi/2, 0, 0, -np.pi]
    elif gate_name == 'Z':
        return [np.pi/2, np.pi,   0, 0]
    elif gate_name == 'S':
        return [np.pi/4, np.pi/2, 0, 0]
    elif gate_name == 'T':
        return [np.pi/8, np.pi/4, 0, 0]
    else:
        return None




""" Method to return a phase shift gate of the desired angle
     
    'params=True'  returns the parameters of the gate
    'params=False' is the default and returns the gate as a matrix 
"""
def phase_shift(theta, param=False):
    parameters = [theta/2, theta, 0, 0]
    
    if param==True:
        return parameters
    else:
        return single_qubit_gate(parameters)




""" Method to return a swap gate, to swap two qubits at the given indices (indexing starts from 0 as usual) 
    The total number of qubits for the whole system must be specified """
def swap(index_one, index_two, total_num_qubits):
    c1 = controlled_unitary(X, index_one, index_two, total_num_qubits)
    c2 = controlled_unitary(X, index_two, index_one, total_num_qubits)
    
    swap_gate = c1@c2@c1
    
    return swap_gate




""" Helper method to be used for the Quantum Fourier Transfrom 
    'R' is a specific instance of the more general phase shift gate """
def R(k):
    theta = np.pi/(2**(k-1))
    return phase_shift(theta)




""" Helper method to be used for the inverse Quantum Fourier Transfrom 
    'R_inv' is a specific instance of the more general phase shift gate """
def R_inv(k):
    theta = -np.pi/(2**(k-1))
    return phase_shift(theta)




""" Method to return the QUANTUM FOURIER TRANSFORM for the desired number of qubits

    NOTE: This is not inherently the most efficient way to simulate a QFT, but the 
          code follows the logic of what one would do with actual quantum logic gates. 
    
    A full explanation of the logic behind this method can be seen in Chapter 5 of 
    "Quantum Computation and Quantum Information" by Nielsen and Chuang
"""
def QFT(number_qubits):
    circuit = tensor_exp(I, number_qubits)
    
    # We apply the initial hadamard gates, followed by the various phase shift gates
    for i in range(number_qubits):
        target  = i
        had     = unitary(H, target, number_qubits) 
        circuit = had@circuit
        
        for k in range(i+1, number_qubits):
            control = k
            gate    = R(1+k-i)     # Gives us R(2), R(3), R(4), ... as k iterates
            controlled_gate = controlled_unitary(gate, target, control, number_qubits)
            circuit = controlled_gate@circuit
    
    # We need to apply the swap gates to complete the Fourier transform
    for k in range(number_qubits//2):
        index_one = k
        index_two = number_qubits-1-k
        kth_swap  = swap(index_one, index_two, number_qubits)
        circuit   = kth_swap@circuit 
    
    return circuit
            



""" Method to return the INVERSE QUANTUM FOURIER TRANSFORM for the desired number of qubits

    NOTE: As with the 'QFT' method, this is not inherently the most efficient way to simulate an 
          inverse QFT, but the code follows the logic of what one would do with actual quantum logic gates. 
          
    A full explanation of the logic behind this method can be seen in Chapter 5 of 
    "Quantum Computation and Quantum Information" by Nielsen and Chuang 
"""
def inverse_QFT(number_qubits):
    circuit = tensor_exp(I, number_qubits)
    
    # We need to apply the swap gates first for the inverse QFT (as opposed to at the end with the regular QFT)
    for k in range(number_qubits//2):
        index_one = k
        index_two = number_qubits-1-k
        kth_swap  = swap(index_one, index_two, number_qubits)
        circuit   = kth_swap@circuit 
    
    # We apply the various phase shift gates, followed by the hadamard
    for i in reversed(range(number_qubits)):
        target  = i
        
        for k in reversed(range(i+1, number_qubits)):
            control = k
            gate    = R_inv(1+k-i)     # Gives us ... , R_inv(4), R_inv(3), R_inv(2) as k iterates
            controlled_gate = controlled_unitary(gate, target, control, number_qubits)
            circuit = controlled_gate@circuit

        had     = unitary(H, target, number_qubits) 
        circuit = had@circuit
    
    return circuit  



  
###############################################################################
######### Section defining basic methods for quantum causal processes #########
###############################################################################

""" Method that returns True if a bipartite process matrix is valid
    Assumes all input and output spaces are of the same dimension
    
    NOTE: This is computationally expensive for large input/output spaces
"""
def bipartite_process_check(W):
    dim = int(len(W)**(1/4)) 
    dA2=dB2= dim
    
    tolerance = 0.0000002

    her_check = check_hermitian(W)    
    eig_check = all(np.round(np.linalg.eigvalsh(W),5)>=0) # checks if all eigenvalues are greater or equal to 0
    trc_check = (abs(trace(W)-int(dA2*dB2))<tolerance)
    
    if (her_check and eig_check and trc_check):                
        b2_W  = xTrace(W,3)
        a2_W  = xTrace(W,1)
        a2b2_W = xTrace(a2_W,3)
        first_check  = (abs((b2_W + a2_W - a2b2_W)-W)<tolerance).all() # Tells us if there are any terms of the A_2B_2, A_1A_2B_2, A_2B_1B_2, A_1A_2B_1B_2 unallowed forms (any term with A_2B_2)
        
        b1b2_W   = xTrace(b2_W,2)
        a2b1b2_W = xTrace(b1b2_W,1)
        second_check = (abs(a2b1b2_W - b1b2_W)<tolerance).all() # Tells us if there are any terms of the A_2 or A_1A_2 unallowed forms

        a1a2_W  = xTrace(a2_W,0)
        a1a2b2_W = xTrace(a1a2_W,3)
        third_check  = (abs(a1a2b2_W - a1a2_W)<tolerance).all() # Tells us if there are any terms of the B_2 or B_1B_2 unallowed forms
        
        if (first_check and second_check and third_check):
            return True
        else:
            return False
        
    else:
        return False




""" Method to define the mathematical tool xW = tensor(I^(x),Tr_x(W))/dim(x) for bipartite process matrices, 
    where x is the input or output space we trace out and factor back in as an identity term.
    
    x=0 indicates A1, x=1 indicates A2, x=2 indicates B1, x=3 indicates B2
    
    We assume all input and output spaces are of equal size """
def xTrace(W,x):
   
    dim    = int(len(W)**(1/4))       # Dimension of each of the input/output spaces
    n      = int(np.log2(dim))        # Number of Hilbert spaces per input/output space
    xrange = list(range(n*x,n*(x+1))) # The indices of Hilbert spaces we will trace out
    W = ptrace_list(W,xrange)         # Tracing out the relevant system
    
    # Traced out A1
    if (x==0):   # tensor I in front, nothing else
        W = tensor( tensor_exp(I,n), W) / dim
    
    # Traced out A2
    elif (x==1): # tensor I in front, swap first and second spaces
        W = tensor( tensor_exp(I,n), W) / dim
        for i in range(n):
            W = swap_spaces(W,i,n+i)
            
    # Traced out B1
    elif (x==2): # tensor I in back, swap third and fourth spaces
        W = tensor(W, tensor_exp(I,n)) / dim
        for i in range(n):
            W = swap_spaces(W,2*n+i,3*n+i)

    # Traced out B2
    elif(x==3):  # tensor I in back, nothing else        
        W = tensor(W, tensor_exp(I,n)) / dim
    
    return W




""" Method to swap two Hilbert spaces in a matrix """
def swap_spaces(W, space_one, space_two):
    n_spaces = int(np.log2(len(W)))  # Number of single qubit Hilbert spaces in our matrix
    
    b0b0 = np.array([[1, 0],
                     [0, 0]])
    b0b1 = np.array([[0, 1],
                     [0, 0]])    
    b1b0 = np.array([[0, 0],
                     [1, 0]])    
    b1b1 = np.array([[0, 0],
                     [0, 1]])
    
    # Identities acting on all the spaces we do not swap
    initial = tensor_exp(I, space_one )
    middle  = tensor_exp(I, (space_two - space_one - 1) )
    final   = tensor_exp(I, (n_spaces - space_two - 1) )

    # Constructing the operator that swaps our two specified Hilbert spaces
    SWAP = tensor(initial, b0b0, middle, b0b0, final) + tensor(initial, b0b1, middle, b1b0, final) +\
           tensor(initial, b1b0, middle, b0b1, final) + tensor(initial, b1b1, middle, b1b1, final)
    
    return SWAP@W@SWAP
    
    
    
    
""" Method to return a larger set by tensoring the elements of a smaller basis set 
    n is the number of times we wish to tensor the basis set with itself """
def tensored_basis_set(basis, n=2):

    new_basis = basis
    for i in range(n-1):
        new_basis = [tensor(x,y) for x in basis for y in new_basis]
    
    return new_basis




""" Method that returns True if the matrix W is of the form A<B or B<A with no claisscal mixture
    
    W is assumed to be an already verified bipartite process
    W can only have input/output spaces of dimension 2 for this method """
def Ao_Bo_process_check(W):
    BoW = tensor(ptrace(W,3),I)/2
    AoW = tensor(I,ptrace(W,1))/2
    AoW = swap_spaces(AoW,0,1)
    
    Bo_check = (abs(W-BoW)<0.000005).all() 
    Ao_check = (abs(W-AoW)<0.000005).all()
    
    if (Ao_check or Bo_check):
        return True
    else:
        return False




""" Method that returns True if S is a valid causal witness
    Only applicable for bipartite systems with input and output dimensions of 2 """
def witness_check(S):
    BoS = tensor(ptrace(S,3),I)/2
    AoS = tensor(I,ptrace(S,1))/2
    AoS = swap_spaces(AoS,0,1)
    eig_check_AoS = all(np.round(np.linalg.eigvalsh(AoS),5)>=0) 
    eig_check_BoS = all(np.round(np.linalg.eigvalsh(BoS),5)>=0)
    her_check = check_hermitian(S)
    
    if (eig_check_AoS and eig_check_BoS and her_check):
        return True
    else:
        return False

        
    
    
""" Method to check if a process or matrix is Hermitian """
def check_hermitian(W):
    return np.all(abs(W-np.conj(W.T))<0.000000001)
    



""" Method to return a random separable process, where the choice of which type can be specified 
    Only 16 by 16 bipartite processes are returned, with each input and output space being 2 by 2 """
def separable_process(choice=-1):
    
    if (choice==-1): # if choice is unspecified, we randomly choose 1 of the 4 possible processes
        choice = np.random.randint(0,4)
        
    if (choice==0):
        W_psi = nonsignalling_nonentangled_process()
    elif (choice==1):
        W_psi = nonsignalling_entangled_process()
    elif (choice==2):
        W_psi = signalling_AtoB_process()
    elif (choice==3):
        W_psi = signalling_BtoA_process()
    
    return W_psi




""" Method to return a random non-signalling non-entangled bipartite process """
def nonsignalling_nonentangled_process():
    Ai = state2dm(separable_state(1))
    Bi = state2dm(separable_state(1))
    W = tensor(Ai,I,Bi,I)
    return W




""" Method to return a random non-signalling entangled bipartite process """
def nonsignalling_entangled_process():
    AiBi = state2dm(entangled_state(2))
    W = tensor(AiBi,I,I)
    W = swap_spaces(W,1,2)
    return W




""" Method to return a random signalling process from Alice to Bob """
def signalling_AtoB_process():
    Ai = state2dm(separable_state(1))
    C = (tensor(X,X)+tensor(Z,Z)-tensor(Y,Y)+tensor(I,I))/2
    W = tensor(Ai,C,I)
    return W
    



""" Method to return a random signalling process from Bob to Alice """
def signalling_BtoA_process():
    Bi = state2dm(separable_state(1))
    C = (tensor(X,X)+tensor(Z,Z)-tensor(Y,Y)+tensor(I,I))/2
    W = tensor(C,Bi,I)
    W = swap_spaces(W,1,3)
    return W




""" Method to return a random bipartite non-separable process matrix
    Only returns matrices of 16 by 16, with each input/output space being 2 by 2 """
def nonseparable_process():
    term1,term2 = nonsep_process_terms()
    sig = [I,X,Y,Z]
        
    n1 = 0
    n2 = 0
    
    while ((n1+n2)<1):
        n1 = rd.uniform(0,1)
        n2 = rd.uniform(0,(1-n1))
        n1 = np.sqrt(n1)
        n2 = np.sqrt(n2)
        
    n1 = ((-1)**np.random.binomial(1,0.5))*n1
    n2 = ((-1)**np.random.binomial(1,0.5))*n2

    t1 = tensor(sig[term1[0]],sig[term1[1]],sig[term1[2]],sig[term1[3]])
    t2 = tensor(sig[term2[0]],sig[term2[1]],sig[term2[2]],sig[term2[3]])
    
    W = (1/4)*(tensor(I,I,I,I) + n1*t1 + n2*t2)
    S = (1/4)*(tensor(I,I,I,I) - np.sign(n1)*t1 - np.sign(n2)*t2)

    if (bipartite_process_check(W) and witness_check(S) and (trace(S@W)<0)):
        return W
    else:
        print("Noncausal process construction error")
        return None

  


""" Method to return the randomised parameters of two causally non-separable process matrix terms, 
    along with their associated coefficients """
def nonsep_process_terms():    
    with_memory    = np.random.binomial(1,0.5)==0 # We decide if one of the terms will be memoryless
    without_memory = not with_memory
    
    AtoB = np.random.binomial(1,0.5)==0           # We decide if term1 will be from A to B, or B to A
    BtoA = not AtoB
    
    term1 = [None]*4
    term2 = [None]*4
    
    # Getting term1
    if (AtoB): # Here, term1 signals from A to B
        term1[0] = 0
        term1[1] = np.random.randint(1,4)
        term1[2] = np.random.randint(1,4)
        term1[3] = 0        
        
        if (with_memory):
            term1[0] = np.random.randint(1,4)

    else:      # In this case, term1 signals from B to A
        term1[0] = np.random.randint(1,4)
        term1[1] = 0        
        term1[2] = 0
        term1[3] = np.random.randint(1,4)       
        
        if (with_memory):
            term1[2] = np.random.randint(1,4)    
    
    
    # Getting term2
    Ai_over_Bi = np.random.binomial(1,0.5)==0  # Determines which input space will match between our two terms
    Bi_over_Ai = not Ai_over_Bi
    lis = [1,2,3]
    
    if (without_memory):
        if (AtoB):   # In this case, as term1 goes from A to B, term2 goes from B to A            
            lis.remove(term1[2])
            
            term2[0] = np.random.randint(1,4)
            term2[1] = 0
            term2[2] = rd.sample(lis,1)[0]
            term2[3] = np.random.randint(1,4)
        
        else:
            lis.remove(term1[0])
            
            term2[0] = rd.sample(lis,1)[0]
            term2[1] = np.random.randint(1,4)
            term2[2] = np.random.randint(1,4)
            term2[3] = 0          
        
    else:
        if (AtoB and Ai_over_Bi):  # indicating that term2 is from B to A
            lis.remove(term1[0])
            
            term2[0] = rd.sample(lis,1)[0]
            term2[1] = 0
            term2[2] = term1[2]
            term2[3] = np.random.randint(1,4)
            
        elif (AtoB and Bi_over_Ai):
            lis.remove(term1[2])
            
            term2[0] = term1[0]
            term2[1] = 0
            term2[2] = rd.sample(lis,1)[0]
            term2[3] = np.random.randint(1,4)
        
        elif (BtoA and Ai_over_Bi):
            lis.remove(term1[0])
            
            term2[0] = rd.sample(lis,1)[0]
            term2[1] = np.random.randint(1,4)
            term2[2] = term1[2]
            term2[3] = 0   
            
        else:
            lis.remove(term1[2])
            
            term2[0] = term1[0]
            term2[1] = np.random.randint(1,4)
            term2[2] = rd.sample(lis,1)[0]
            term2[3] = 0   
    
    return term1,term2
        
    