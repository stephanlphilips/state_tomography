import numpy as np

def generate_paulis():
	pauli_dict = dict()
	pauli_dict['I'] = np.matrix([[1,0],
				   [0,1]], dtype = np.complex)
	pauli_dict['X'] = np.matrix([[0,1],
				   [1,0]], dtype = np.complex)
	pauli_dict['Y'] = np.matrix([[0 ,-1j],
				   [1j,0  ]], dtype = np.complex)
	pauli_dict['Z'] = np.matrix([[1,0],
				   [0,-1]], dtype = np.complex)
	
	pauli_name_list = list(pauli_dict.keys())

	return pauli_dict, pauli_name_list

def generate_pauli_names(n_qubits, pauli_list=None):
	'''
	generates a list with all possiblities of paulis for a given number of qubits
	'''
	if pauli_list == None:
		pauli_list = generate_paulis()[1] 
	
	pauli_s = []
	pauli_name_list = generate_paulis()[1] 
	for i in range(len(pauli_list)):
		for j in range(4):
			pauli_s.append(pauli_list[i]+pauli_name_list[j])

	if n_qubits == 2:
		return pauli_s
	else:
		return generate_pauli_names(n_qubits-1, pauli_s)

def pauli_to_matrix(pauli_list):
	pauli_dict = generate_paulis()[0] 
	mat_tmp = pauli_dict[pauli_list[0]]
	for mat in pauli_list[1:]:
		mat_tmp = np.kron(mat_tmp, pauli_dict[mat])
	return mat_tmp

def generate_pauli_matrix_list(mat_list):
	pauli_dict = generate_paulis()[0] 
	mats = list()
	for mat_list_single in mat_list:
		mats.append(pauli_to_matrix(mat_list_single))
	return mats

def pauli_matrix_list_to_big_mat(matrix_list):
	mat = np.zeros((len(matrix_list), len(matrix_list)), dtype=np.complex)

	for i in range(len(matrix_list)):
		mat[i] = matrix_list[i].reshape((len(matrix_list), ))

	return mat.T

