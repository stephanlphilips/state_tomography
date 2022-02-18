from routines.measurements.compile_instructions import compile_measurements

from dataclasses import dataclass
import numpy as np

def compile_witnesses(witnesses_operators, qubits_used, measurement_operators):
	'''
	compiles a set op witness operators to operators that can be measured on the sample.
	Args :
		witnesses_operators (measurment_operator) : list of operators that should be measured.
		qubits_used (list) : list with the qubit number to which the measurement operators correspond
		measurement_operators (list<str>) : list of measurement operators per qubit -- available for measurement on the sample.
	'''
	m = compile_measurements(witnesses_operators, qubits_used, measurement_operators)
	return m.compile('witnesses readout instruction')

if __name__ == '__main__':
	from routines.witnesses.witness_generator import optimal_witness, stabilizer_witness, mermin_witness

	# 4 qubits example
	qubits = [2,3,4,5]
	W = stabilizer_witness(len(qubits))
	m_operator = ['Z', 'Z', 'Z', 'Z']
	
	c = compile_witnesses(W, qubits, m_operator)
	print(c)

	# 5 qubit example with a ZZ readout operator on qubit 1 and 2
	qubits = [1,2,3,4,5,6]
	W = stabilizer_witness(len(qubits))
	m_operator = ['Z_Z2', 'Z_Z1', 'Z', 'Z', 'Z', 'Z']

	c = compile_witnesses(W, qubits, m_operator)
	print(c)