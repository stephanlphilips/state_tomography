from routines.measurements.compile_instructions import compile_measurements

from dataclasses import dataclass
import numpy as np

def compile_state_tomography(m_operators, qubits_used, measurement_operators):
	'''
	compiles a set op witness operators to operators that can be measured on the sample.
	Args :
		m_operators (measurment_operator) : list of operators that should be measured.
		qubits_used (list) : list with the qubit number to which the measurement operators correspond
		measurement_operators (list<str>) : list of measurement operators per qubit -- available for measurement on the sample.
	'''
	m = compile_measurements(m_operators, qubits_used, measurement_operators)
	return m.compile('State tomography operator')

if __name__ == '__main__':
	from routines.state_tomography.state_tomography_generator import state_tomography

	# 3 qubits example
	qubits = [2,3,4]
	M = state_tomography(len(qubits))
	m_operator = ['Z', 'Z', 'Z']
	
	c = compile_state_tomography(M, qubits, m_operator)
	print(c.name)

	# # two qubit example with Z,Z operator for measurement
	# qubits = [1,2]
	# M = state_tomography(len(qubits))
	# m_operator = ['Z', 'Z']

	# c = compile_state_tomography(M, qubits, m_operator)
	# print(c)

	# # two qubit example with ZZ operator for measurement
	# qubits = [1,2]
	# M = state_tomography(len(qubits))
	# m_operator = ['Z_Z2', 'Z_Z1']

	# c = compile_state_tomography(M, qubits, m_operator)
	# print(c)