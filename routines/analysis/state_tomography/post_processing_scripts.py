import matplotlib.pyplot as plt
import numpy as np

from qutip.visualization import matrix_histogram, matrix_histogram_complex
from scipy.linalg import sqrtm

def plot_DM(DM):
	n_qubits = int(np.log2(DM.shape[0]))
	xlabel = []
	for i in range(DM.shape[0]):
		xlabel.append(''.join(list(f'{i:08b}')[-n_qubits:]))

	return matrix_histogram_complex(np.array(DM), xlabel, xlabel)

def calc_concurrence(DM):
	YY = np.matrix( [[0,0,0,-1],
				[0,0,1,0],
				[0,1,0,0],
				[-1,0,0,0]])
	rho_tilda = YY@np.conjugate(DM)@YY 
	R = sqrtm(sqrtm(DM)@rho_tilda@sqrtm(DM))
	w, v= np.linalg.eig(R)
	w = np.real(w)
	lambdas = np.sort(w)

	return max(0.0, lambdas[3]-lambdas[2]-lambdas[1]-lambdas[0])

def calc_state_fidelity(DM, expected_state):
	psi = expected_state.H@expected_state

	return np.real(np.trace(sqrtm(sqrtm(psi)@DM@sqrtm(psi)))**2)

def calc_state_fidelity2(DM, expected_state):
	return expected_state@DM@expected_state.T