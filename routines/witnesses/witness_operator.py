from routines.utility import generate_pauli_names, generate_pauli_matrix_list, pauli_matrix_list_to_big_mat
from routines.measurements.m_operator import measurment_operator
from dataclasses import dataclass

import numpy as np

def generate_operators_from_witness(witness):
	Witness_linear = np.reshape(witness, (witness.size, )).T

	pauli_names = generate_pauli_names(int(np.log2(witness.shape[0])))
	pauli_matrix_list = generate_pauli_matrix_list(pauli_names)
	pauli_big_mat = pauli_matrix_list_to_big_mat(pauli_matrix_list)

	# conververt everything into vectors and solve by elimintion (this still works well for 6 qubits)  
	solution = (np.linalg.solve(pauli_big_mat, Witness_linear))

	args = np.where(solution!=0)

	W = witness_operator()
	for i in args[0]:
		W.add_operator(pauli_names[i],solution[i].real)
	return W

@dataclass
class witness_operator_single:
	prob : float
	name : str
	_matrix : np.ndarray = None

	@property
	def matrix(self):
		if self._matrix is None:
			self._matrix = generate_pauli_matrix_list([self.name])[0]
		return self._matrix

class witness_operator(measurment_operator):
	def add_operator(self, name, probability):
		if round(float(probability),10) != 0:
			self.operators.append(witness_operator_single(float(probability), name))

	def calculate_witness(self, density_matrix):
		W = 0
		for witness in self.operators:
			W += witness.prob*np.trace(density_matrix*witness.matrix)
		return W
	
	def __repr__(self):
		if len(self.operators) == 0:
			return 'This winess is empty'

		descr = 'Operators for witness :: \n'
		for w in self.operators:
			descr += f'\t'
			if w.prob > 0:
				descr += f' '
			descr += f'{round(w.prob,5)} \t{w.name} \n'
		return descr
