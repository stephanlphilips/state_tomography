from routines.witnesses.witness_generator import optimal_witness, stabilizer_witness, mermin_witness
from routines.witnesses.witness_compiler import compile_witnesses

from routines.analysis.get_expectation_value import expectation_value_calculator
from routines.analysis.plot_expectation_value import plot_expectation_values

from routines.utility import pauli_to_matrix

import numpy as np


class W_data_processor (expectation_value_calculator):
    def __init__(self, ds, qubits_used, m_operators, witness_type='optimal', state=None,):
        '''
        Args :
            ds (dataset) : dataset to be used to extract the operators from
            qubits_used (list) : list with the number of the qubits that are used in the state
            m_operators (list<str>) : list of measurement operators per qubit, in case of combined measurement, refer to the basis and other qubit number.
            witness_type (str) : type of witness to be used (optimal, stabilizer, mermin)
            state (np.matrix) : state vector if not using the default state
        '''
        witness_generator = None
        if witness_type == 'optimal':
            witness_generator = optimal_witness
        elif witness_type == 'stabilizer':
            witness_generator = stabilizer_witness
        elif witness_type == 'mermin':
            witness_generator = mermin_witness
        else:
            raise ValueError('Witness type not recognized.')

        self.W = witness_generator(len(qubits_used), state)
        self.W_c = compile_witnesses(self.W, qubits_used, m_operators)
        self.state = state
        super().__init__(ds, self.W_c, qubits_used, m_operators)

    def plot_expectation_values(self, iteration = 0, state = None):
        '''
        plot the expected expectation values for a given state vector

        Args:
            iteration (int): interation number in the experiment
            state (np.ndarray) : state vector
        '''
        if state is None:
            state=self.state

        if state is not None:
            state = np.matrix(state)
            if state.shape[0] == 1:
                state = state.H

        plot_expectation_values(state, self.calc_expectation_values(iteration))

    def calculate_witness(self, iteration = 0):
        '''
        calculate the witness operator

        Args:
            iteration (int) : iteration number in the experiment
        '''
        expectation_values = self.calc_expectation_values(iteration)
        return self.__calc_witness(expectation_values)

    def calculate_witness_from_DM(self, DM):
        '''
        Calculate witness operators from a given density matrix

        Args:
            DM (np.matrix) : density matrix of the system
        '''
        expectation_values = self.calc_expectation_values(0)

        for name, value in expectation_values.items():
            expectation_values[name] = np.trace(DM@pauli_to_matrix(name)).real

        return self.__calc_witness(expectation_values)

    def __calc_witness(self, expectation_values):
        W  = 0
        for operator in self.W:
            W += operator.prob*expectation_values[operator.name]

        if W < 0:
            print(f'Good news, state is entangled;\n\tWitness = {round(W,3)}')
        else:
            print(f'Meh, this looks like a classical state\n\tWitness = {round(W,3)}')
        return W