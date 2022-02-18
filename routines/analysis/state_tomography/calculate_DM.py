from routines.state_tomography.state_tomography_generator import state_tomography
from scipy.linalg import sqrtm, cholesky

import numpy as np
import scipy as sp

from numpy import linalg as la

def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.conj().T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.conj().T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2
    A3 = (A2 + A2.conj().T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False

def inversion(expectation_values):
    '''
    Get the density matrix by summing up the expectation values.

    Args:
        expectation_values (dict) : dict with key's of the measurement operator with measured expectation
    '''
    n_qubits = int(np.log2(np.sqrt(len(expectation_values))))
    dim_DM = 2**n_qubits

    operators = state_tomography(n_qubits)
    dm_matrix = np.matrix(np.zeros([dim_DM,dim_DM], dtype =np.complex))

    for operator in operators:
        dm_matrix += expectation_values[operator.name]*operator.matrix/dim_DM

    # ensure postive definite;
    dm_matrix_p =  nearestPD(dm_matrix)
    
    # normalize matrix
    return dm_matrix_p/np.trace(dm_matrix_p)

def MLE(expectation_values):
    '''
    Run a Maximum likelyhood estimation to fit a density matrix to the measured expectation values

    Args:
        expectation_values (dict) : dict with key's of the measurement operator with measured expectation
    '''
    n_qubits = int(np.log2(np.sqrt(len(expectation_values))))

    dm_matrix_guess = inversion(expectation_values)

    t_matrix_guess = cholesky(dm_matrix_guess, lower=True)
    DM_vect_guess = t_matrix_to_vect(t_matrix_guess, 2**n_qubits)

    # ensure correct ordering of operators -- expectation values.
    operators = state_tomography(n_qubits)
    L_measure = np.zeros( (len(operators),))
    for idx, operator in enumerate(operators):
        L_measure[idx] = expectation_values[operator.name]

    popt, pcov = sp.optimize.curve_fit(calc_pauli_vector(n_qubits), operators, L_measure , p0=DM_vect_guess, method='trf')#, bounds = ([-1, ]*DM_vect_guess.size, [1,]*DM_vect_guess.size))

    return vect_to_DM(popt, 2**n_qubits)

def calc_pauli_vector(n_qubits):
    dim_DM = 2**n_qubits

    def calc_pauli_vectors(operators,*DM_vect):
        dm = vect_to_DM(DM_vect, dim_DM)
        L = np.zeros( (len(operators),))

        for idx, operator in enumerate(operators):
            L[idx] = np.trace(dm@operator.matrix).real

        return L
    return calc_pauli_vectors

def vect_to_DM(vect, dim_DM):
    t_mat = vect_to_t_matrix(vect, dim_DM)
    return t_mat@t_mat.H/np.trace(t_mat.H@t_mat)

def DM_to_vect(DM, dim_DM):
    t_matrix = cholesky(DM, lower=True)
    DM_vect_guess = t_matrix_to_vect(t_matrix, dim_DM)
    return t_matrix_to_vect(t_matrix, dim_DM)

def t_matrix_to_vect(t_matrix, dim_DM):
    n = t_matrix.shape[0]
    vect = np.zeros( int(((n*(n+1)/2) +  (n*(n-1)/2)),))

    vect[0:int(n*(n+1)/2)] = t_matrix.real[np.nonzero(np.tri(n, n,0))]
    vect[int(n*(n+1)/2):] = t_matrix.imag[np.nonzero(np.tri(n, n,-1))]

    return vect

def vect_to_t_matrix(t_matrix_vect, dim_DM):
    t_mat = np.matrix(np.zeros((dim_DM,dim_DM), dtype=np.complex))
    t_mat[np.nonzero(np.tri(dim_DM, dim_DM,-1))] = t_matrix_vect[int(dim_DM*(dim_DM+1)/2):]
    t_mat *= 1j
    t_mat[np.nonzero(np.tri(dim_DM, dim_DM,0))] += t_matrix_vect[0:int(dim_DM*(dim_DM+1)/2)]
    
    return t_mat

if __name__ == '__main__':
    basis = ['II', 'IX', 'IY', 'IZ', 'XI', 'XX', 'XY', 'XZ', 'YI', 'YX', 'YY', 'YZ', 'ZI', 'ZX', 'ZY', 'ZZ']
    m_result = [ 1.00000000e+00, 1.04324968e-01, 9.78252690e-02, 7.39980828e-02, 4.96362858e-02, 9.71151524e-01, 7.56660058e-02, -1.46070608e-01, 6.92165940e-02, -1.65393723e-01, 9.39160745e-01, -1.17851700e-02,8.92933803e-02, -1.41364439e-01, -2.05067815e-05, -1.00924410e+00]
    
    from routines.analysis.state_tomography.post_processing_scripts import plot_DM, calc_concurrence, calc_state_fidelity, calc_state_fidelity
    import matplotlib.pyplot as plt
    
    expectation_values = {}
    for i,j in zip(basis, m_result):
        expectation_values[i] = j

    mat1 = inversion(expectation_values)
    mat2 = MLE(expectation_values)

    plot_DM(mat1)
    plot_DM(mat2)

    plt.show()