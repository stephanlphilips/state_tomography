from routines.analysis.state_tomography.post_processing_scripts import plot_DM, calc_concurrence, calc_state_fidelity, calc_state_fidelity
from routines.analysis.get_expectation_value import expectation_value_calculator
from routines.analysis.state_tomography.calculate_DM import inversion, MLE 
from routines.state_tomography.state_tomography_compiler import compile_state_tomography
from routines.state_tomography.state_tomography_generator import state_tomography
from routines.analysis.plot_expectation_value import plot_expectation_values as plot_exp
from routines.witnesses.witness_generator import optimal_witness, stabilizer_witness, mermin_witness

class ST_data_processor (expectation_value_calculator):
    def __init__(self, ds, qubits_used, m_operators):
        '''
        Args :
            ds (dataset) : dataset to be used to extract the operators from
            qubits_used (list) : list with the number of the qubits that are used in the state
            m_operators (list<str>) : list of measurement operators per qubit, in case of combined measurement, refer to the basis and other qubit number.
        '''
        st = state_tomography(len(qubits_used))
        st_c = compile_state_tomography(st, qubits_used, m_operators)

        super().__init__(ds, st_c, qubits_used, m_operators)

    def get_DM(self, iteration=0, method = 'MLE'):
        '''
        calculate the density matrix

        Args:
            iteration (int): interation number in the experiment
            method (str): method to use (MLE or inversion)
        '''
        if method == 'MLE':
            return MLE(self.calc_expectation_values(iteration))
        return inversion(self.calc_expectation_values(iteration))

    def plot_expectation_values(self, iteration = 0, state = None):
        '''
        plot the expected expectation values for a given state vector

        Args:
            iteration (int): interation number in the experiment
            state (np.ndarray) : state vector
        '''
        plot_exp(state, self.calc_expectation_values(iteration))


    def plot_DM(self, iteration=0, method = 'MLE'):
        DM = self.get_DM(iteration, method)
        plot_DM(DM)
        return DM
    
    def calculate_witness(self, iteration=0, state=None, witness_type='optimal'):

        witness_generator = None
        if witness_type == 'optimal':
            witness_generator = optimal_witness(len(self.qubits_used), state)
        elif witness_type == 'stabilizer':
            witness_generator = stabilizer_witness(len(self.qubits_used), state)
        elif witness_type == 'mermin':
            witness_generator = mermin_witness(len(self.qubits_used), state)
        else:
            raise ValueError('Witness type not recognized.')

        self.W = witness_generator
        
        W  = 0
        expectation_values = self.calc_expectation_values(iteration)
        for operator in self.W:
            W += operator.prob*expectation_values[operator.name]

        if W < 0:
            print(f'Good news, state is entangled;\n\tWitness = {round(W,3)}')
        else:
            print(f'Meh, this looks like a classical state\n\tWitness = {round(W,3)}')
        return W