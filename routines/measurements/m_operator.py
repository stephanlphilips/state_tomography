from routines.utility import generate_pauli_matrix_list
from dataclasses import dataclass
import numpy as np

@dataclass
class operator_single:
	name : str
	_matrix : np.ndarray = None

	@property
	def matrix(self):
		if self._matrix is None:
			self._matrix = generate_pauli_matrix_list([self.name])[0]
		return self._matrix

class measurment_operator:
	def __init__(self):
		self.operators = []

	def add_operator(self, operator):
		self.operators.append(operator_single(operator))

	def __len__(self):
		return len(self.operators)

	def __iter__(self):
		self.n = 0
		return self
	
	def __next__(self):
		if self.n >= len(self):
			raise StopIteration
		self.n+=1
		return self.operators[self.n-1]

	def __getitem__(self, i):
		return self.operators[i]

	def __repr__(self):
		if len(self.operators) == 0:
			return 'No measurement operators present'

		descr = 'Operators to be measured :: \n'
		for w in self.operators:
			descr += f'\t{w.name} \n'
		return descr