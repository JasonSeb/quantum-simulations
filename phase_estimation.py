"""
# Author  : Jason Connie
# Created : July 2022

This code simulates the circuit a quantum computer would use for phase estimation.

Please feel free to set the phase 'phi' and the precision 't' as desired.
Though keep in mind having more than 8 qubits of precision can rapidly become computationally costly, 
as simulating qubits and their dynamics is an exponentially expensive task.

Phase estimation is an important protocol used within Shor's algorithm, in quantum 
solutions to the discrete log problem, and in a few other problems. All such solutions
are rooted in the Quantum Fourier Transform. For those who are interested, Chapter 5 
of Nielsen and Chuang's "Quantum Computation and Quantum Information" is the definitive 
introduction to the topic.

"""

# We import the necessary modules, and instantiate a few important quantum gates and basis vectors
import qcausal as q
import numpy   as np

I = q.I   # Identity gate
H = q.H   # Hadamard gate

b0 = np.array([[1],[0]])   # computational basis vector |0)
b1 = np.array([[0],[1]])   # computational basis vector |1)
bp = (1/np.sqrt(2))*np.array([[1],[1]])  # basis vector |+)
bm = (1/np.sqrt(2))*np.array([[1],[-1]]) # basis vector |-)




""" Method to construct the phase estimation circuit for a SINGLE qubit gate U
    
    The mathematical details, accompanied by an illustrative diagram, can be 
    seen in Chapter 5.2 of "Quantum Computation and Quantum Information" 
"""
def phase_estimation_circuit(U, num_freg_qubits):
    total_num_qubits = num_freg_qubits + 1
    
    had_fr   = q.tensor_exp(H, num_freg_qubits) # Hadamard gates to be applied to the qubits within the first register
    circuit  = q.tensor(had_fr, I)
    
    for i in range(num_freg_qubits):
        control = num_freg_qubits-1-i
        target  = num_freg_qubits
        gate    = np.linalg.matrix_power(U, 2**i) # Raising U to the necessary power
    
        controlled_gate = q.controlled_unitary(gate, target, control, total_num_qubits) # The controlled application of the exponentiated U
        circuit         = controlled_gate@circuit
    
    inv_qft = q.inverse_QFT(num_freg_qubits) # The inverse Quantum Fourier Transform to be applied to the first register qubits 
    circuit = q.tensor(inv_qft, I)@circuit
    
    return circuit




""" Method that takes a binary string b0b1b2... and returns the fraction 0.b0b1b2... in base-10 form
    Where 0.b0b1b2... = b0/(2**1) + b1/(2**2) + b2/(2**3) + ... 
"""
def bit_string_to_fraction(bit_str):
    if (bit_str[1]=='b'):    # If the '0b' prefix is at the beginning of the string, we remove it
        bit_str = bit_str[2:]
    
    result = 0
    
    for i in range(len(bit_str)):
        result += int(bit_str[i])/2**(i+1)
    
    return result




if __name__=="__main__":
    # A phase value 'phi' gives us unique informartion about an eigenvalue of a quantum logic gate, as eig_val = np.exp(phi*1j*2*pi)
    # The range of our phase is 0 <= phi < 1
    phi = 0.165151
    
    
    # Any phase (given as 'phi') can be expressed as the sum of fractional powers of 2, with numerators of 1 and 0
    # phi = 1/2 + 0/4 + 1/8 + 0/16 + 0/32 + 1/64 + 1/128 is an example that can be represented with 7 bits in this form
    # In general, the phase estimation algorithm returns the best t-bit approximation of 'phi'
    # NOTE: Don't go beyond 8 qubits unless you have patience. Simulating qubits is exponentially expensive, and even 9 qubits can take more than a couple seconds
    t = 7
    
    
    # Though it is usually unknown to those performing the algorithm, in this simulation we instantiate the single qubit gate U ourselves.
    # The first eigenvector ('eig_vec') and its related eigenvalue correspond to the phase 'phi'.
    # In this simulation, the phase of the second eigenvalue does not concern us and can be set to anything.
    U = np.exp(2*np.pi*1j*phi)*bp@np.conj(bp.T) - np.exp( 1j*2*np.pi*np.random.uniform(0,1) )*bm@np.conj(bm.T) 
    eig_vec = bp 
    
    
    
    # We can then give the necessary eigenvector and the gate U to another individual who does not know the value of 'phi',
    # and ask them to figure out the best t-bit approximation
    circuit = phase_estimation_circuit(U, t)
    
    input_state     = q.tensor( q.tensor_exp(b0, t), eig_vec)
    output_state    = circuit@input_state
    freg_output_rho = q.ptrace( q.state2dm(output_state), t) # We trace out the second register qubits after the circuit, and so need the density matrix formalism
    
    
    
    # The basis vector with the highest probability will represent the best t-bit approximation of 'phi'
    # If we get |1010011) for example, that would correspond to the binary fraction 0.1010011
    outcomes    = q.measure(freg_output_rho)
    best_approx = np.argmax(outcomes) # The index (basis-value) of the measurement outcome with the highest probability
    basis_vec   = str(bin(best_approx))[2:].zfill(t)
    
    print("The actual value for phi is ",phi,", and ",t," bits of precision were desired.\n", sep="")
    print("The phase estimation algorithm results in the outcome |",basis_vec,"), which corresponds to ",bit_string_to_fraction(basis_vec),"\n", sep="")
    print("The probability of this successful outcome is",np.round(max(outcomes),8))
    
    
