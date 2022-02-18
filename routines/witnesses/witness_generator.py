from routines.witnesses.witness_operator import generate_operators_from_witness
from routines.utility import pauli_to_matrix
from itertools import permutations

import numpy as np

def optimal_witness(n_qubits, state=None):
	'''
	Optimal witness for the GHZ state.
	'''
	if state is None:
		state = np.matrix([1]*(2**n_qubits), dtype=np.complex)/np.sqrt(2)
		state[0,1:-1] = 0

	witness = np.eye(state.size)/2 - state.H*state
	return generate_operators_from_witness(witness)

def stabilizer_witness(n_qubits, state=None):
	'''
	Stabilizer based witness for the GHZ state (similar to optimal one, less measurements, but less noise resilience)
	'''
	if state is not None:
		raise ValueError('inputting state :: not supported')
	
	witness = 2*pauli_to_matrix('I'*n_qubits)
	witness -= pauli_to_matrix('X'*n_qubits)
	
	M = pauli_to_matrix('I'*n_qubits)
	for i in range(n_qubits-1):
		g_k = list('I'*n_qubits)
		g_k[i] = 'Z'
		g_k[i+1] = 'Z'
		M *= (pauli_to_matrix(g_k) + pauli_to_matrix('I'*n_qubits))/2
	
	witness -= 2*M 
	return generate_operators_from_witness(witness)

def mermin_witness(n_qubits, state=None):
	'''
	Witness for GHZ states based on the mermin equality.
	'''
	if state is not None:
		raise ValueError('inputting state :: not supported')
	
	witness = 2**(n_qubits-2)*pauli_to_matrix('I'*n_qubits)

	sign = -1
	for i in range(int((n_qubits+2)/2)):
		oper = np.asarray(list('X'*n_qubits))
		oper[0:2*i] = 'Y'
		oper_list = list(set([i for i in permutations(oper)]))
		O = pauli_to_matrix(''.join(oper_list[0]))
		for i in oper_list[1:]:
			O+= pauli_to_matrix(''.join(i))
		witness += sign*O
		sign *= -1

	return generate_operators_from_witness(witness)

def check_witness(dm_noise):
	n_qubits = int(np.log2(dm_noise.shape[0]))

	W1 = optimal_witness(n_qubits)
	W2 = stabilizer_witness(n_qubits)
	W3 = mermin_witness(n_qubits)

	state = np.matrix([1]*(2**n_qubits), dtype=np.complex)/np.sqrt(2)
	state[0,1:-1] = 0

	p_noise = np.linspace(0, 1)
	witness_outcome1 = np.empty(p_noise.size)
	witness_outcome2 = np.empty(p_noise.size)
	witness_outcome3 = np.empty(p_noise.size)

	for i in range(p_noise.size):
		DM = (1-p_noise[i])*state.H*state + p_noise[i]*dm_noise
		witness_outcome1[i] = W1.calculate_witness(DM)
		witness_outcome2[i] = W2.calculate_witness(DM)
		witness_outcome3[i] = W3.calculate_witness(DM)

	import matplotlib.pyplot as plt
	# plot witness for the different operators, where the state is smaller than 0, the state is entangled.
	plt.plot(p_noise, witness_outcome1, label='Generic')
	plt.plot(p_noise, witness_outcome2, label='Stabilizer')
	plt.plot(p_noise, witness_outcome3, label='Mermin')

	min_val = np.min([0,np.min(witness_outcome1), np.min(witness_outcome2), np.min(witness_outcome3)])
	plt.fill_between((0, 1), (min_val, min_val), interpolate=True, color='red', alpha=0.2, label='Entangled area')
	plt.xlabel('p noise (%)')
	plt.ylabel('Expectation value')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	# print human readable version of the witness
	print(optimal_witness(3))
	
	# check textbook example to see if the witness goes to the right values
	check_witness(np.eye(2**3)/8)

	# flip the phase
	state = np.matrix([1]*(2**3), dtype=np.complex)/np.sqrt(2)
	state[0,1:-1] = 0
	state[0,-1] *= -1

	check_witness(state.H*state)