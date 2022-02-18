from routines.utility import pauli_to_matrix

import matplotlib.pyplot as plt
import numpy as np

def plot_expectation_values(state, expectation):
    a = list(expectation.values())
    x = [i for i,j in enumerate(a)]

    plt.figure()
    plt.title('Expectation values')
    plt.bar(x, a, linewidth=0, alpha=0.5)

    if state is not None:
        ideal_value = [0]*len(a)
        for idx, operator in enumerate(expectation.keys()):
            ideal_value[idx] = float(np.real(state@pauli_to_matrix(operator)@state.H))
        plt.bar(x, ideal_value, edgecolor='black',  fill=False,linewidth=1)
    
    plt.ylabel('Expectation value')
    x_pos = np.arange(len(a))
    x_label = list(expectation.keys())
    plt.xticks(x_pos, labels=x_label)
    plt.xticks(rotation='vertical')

    plt.show()

if __name__ == '__main__':
    basis = ['II', 'IX', 'IY', 'IZ', 'XI', 'XX', 'XY', 'XZ', 'YI', 'YX', 'YY', 'YZ', 'ZI', 'ZX', 'ZY', 'ZZ']
    m_result = [ 1.00000000e+00, 1.04324968e-01, 9.78252690e-02, 7.39980828e-02, 4.96362858e-02, 9.71151524e-01, 7.56660058e-02, -1.46070608e-01, 6.92165940e-02, -1.65393723e-01, 9.39160745e-01, -1.17851700e-02,8.92933803e-02, -1.41364439e-01, -2.05067815e-05, -1.00924410e+00]

    expect = {}
    for i,j in zip(basis, m_result):
        expect[i] = j

    # plot_expectation_values(None, expect)
    print(expect)
    plot_expectation_values(np.matrix([[0,1,1,0]])/np.sqrt(2), expect)