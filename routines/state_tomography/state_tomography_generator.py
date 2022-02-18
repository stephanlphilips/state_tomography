from routines.utility import generate_pauli_names
from routines.measurements.m_operator import measurment_operator

def state_tomography(n_qubits):
	pauli_s = generate_pauli_names(n_qubits)

	m = measurment_operator()
	for pauli in pauli_s:
		m.add_operator(pauli)

	return m

if __name__ == '__main__':
	s = state_tomography(3)
	print(s)