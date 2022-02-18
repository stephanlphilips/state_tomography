from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class monte_carlo_measurement:
    expectation : float
    n_measurements : int

    @property
    def measure(self):
        if self.n_measurements == 0:
            return self.expectation
        if self.expectation > 1:
            exp = 1
            print(f'Warning expecation value out of range! ({self.expectation})')
        elif self.expectation < -1:
            exp = -1
            print(f'Warning expecation value out of range! ({self.expectation})')
        else:
            exp = self.expectation
        return np.random.binomial(self.n_measurements, exp/2 + 0.5)/self.n_measurements*2-1
    
@dataclass
class readout_properties:
    operator : str
    visibility : tuple
    flip : bool

    @property
    def fid_matrix(self):
        F_down, F_up = self.visibility
        M = np.matrix([ [F_down,  1-F_up],
                        [1-F_down,F_up]])

        if self.flip == True:
            return np.matrix([[0,1],[1,0]])*M

        return M

class readout_properties_mgr:
    def __init__(self):
        self.operators_w_visibilty = []

    def add(self, operator, visibility, flip):
        v = readout_properties(operator, visibility, flip)
        self.operators_w_visibilty.append(v)

    def __getitem__(self, name):
        for prop in self.operators_w_visibilty:
            if prop.operator == name :
                return prop

        print(f'Warning :: operator {name} not found, asuming 100% readout fidelity')
        return readout_properties(name, (1,1), False)

def decompose_measurement_operator(operator, qubits, m_operators_sample):
    fundamental_operators = []
    measurement_names = []

    operator = list(operator)
    to_skip = []
    for i in range(len(list(operator))):
        if operator[i] != 'I' and i not in to_skip:
            oper = ['I']*len(qubits)
            oper[i] = 'Z'
            if m_operators_sample[i] == 'Z':
                fundamental_operators.append(''.join(oper))
                measurement_names.append(f'read{qubits[i]}')
            else:
                _, co_measurement = m_operators_sample[i].split('_')
                co_measurement = int(''.join(j for j in co_measurement if j.isdigit()))

                if operator[qubits.index(co_measurement)] != 'I':
                    oper[qubits.index(co_measurement)] = 'Z'

                to_skip.append(qubits.index(co_measurement))
                fundamental_operators.append(''.join(oper))
                measurement_names.append(f'read{qubits[i]}')
    return fundamental_operators, measurement_names

def get_state(state, data):
    if len(data) == 1:
        return np.where(data[0] == int(state[0]))[0].size
    else:
        idx = np.where(data[0] == int(state[0]))[0]
        new_data = []
        for i in range(1,len(data)):
            new_data.append(data[i][idx])
        return get_state(state[1:], new_data)

def get_probablity_vector(ds, i, reads):
    if len(reads) == 0: #identity case
        return np.matrix([1, 0]).T, 0

    output_vector = [0]*int(2**len(reads))

    read_data = []
    for read in reads: 
        read_data.append(ds[f'State {read}'][i]()[np.where(ds['mask']()[i]==1)])
    n_meas = read_data[0].size
    for i in range(len(output_vector)):
        output_vector[i] = get_state( list(f'{i:08b}')[-len(reads):], read_data)/n_meas
    return np.matrix(output_vector).T, n_meas

def convert_probability_vector_to_expectation_value(expectation, m_operators, readout_prop):
    F = np.matrix([1])
    for oper in m_operators:
        vis_fix = readout_prop[oper].fid_matrix
        F = np.kron(F, vis_fix)
    
    F = np.matrix(np.eye(2)) if F.size == 1 else F
    m_vector =  np.array(np.linalg.inv(F)*expectation).flatten()

    M = np.matrix([1])
    Z_measuerement = np.array([1,-1])
    for i in range(len(m_operators)):
        M = np.kron(M, Z_measuerement)

    return np.sum(m_vector[np.where(M == 1)[1]]) - np.sum(m_vector[np.where(M == -1)[1]])

class expectation_value_calculator:
    def __init__(self, ds, measured_operators, qubits_used = [1,2,3,4,5,6], m_operators = ['Z_Z2', 'Z_Z1', 'Z', 'Z', 'Z_Z6', 'Z_Z5']):
        '''
        Args :
            ds (dataset) : dataset to be used to extract the operators from
            measured_operators (tbd) : list of operators that have been measured on the sample
            qubits_used (list) : list with the number of the qubits that are used in the state
            m_operators (list<str>) : list of measurement operators per qubit, in case of combined measurement, refer to the basis and other qubit number.
        '''
        self.ds = ds
        self.measured_operators = measured_operators
        self.qubits_used = qubits_used
        self.m_operators = m_operators
        self.readout_properties_mgr = readout_properties_mgr()
    
    def add_readout_properties(self, operator, visibility, flip = False):
        '''
        set visibilites for the different measurement operators on the sample
        Args :
            operator (str) : operator for a certain measurement (e.g. ZZIIII)
            visibilites (tuple) : visibility spin down
            flip (bool) : true if the expectation value is flipped, e.g. for a S/T measurment 0 is antiparallel
        '''
        self.readout_properties_mgr.add(operator, visibility, flip)

    def calc_expectation_values(self, iteration=0, monte_carlo=False):
        m_op = self.measured_operators.flatten()
        exp_values = dict()
        for idx, operator_instruction in enumerate(m_op):
            fundamental_operators, measurement_names = decompose_measurement_operator(operator_instruction.operators[0].name, self.qubits_used, self.m_operators)
            probability_vector,n_meas = get_probablity_vector(self.ds, operator_instruction.m_id + iteration*len(self.measured_operators), measurement_names)
            expectation = convert_probability_vector_to_expectation_value(probability_vector, fundamental_operators, self.readout_properties_mgr)
            exp_values[operator_instruction.operators[0].name] = expectation
            if monte_carlo == True:
                exp_values[operator_instruction.operators[0].name] = monte_carlo_measurement(expectation, n_meas)

        return exp_values

if __name__ == '__main__':
    from core_tools.data.ds.data_set import load_by_uuid

    from routines.state_tomography.state_tomography_generator import state_tomography
    from routines.state_tomography.state_tomography_compiler import compile_state_tomography
    
    qubits = [1,2,3]
    m_operator = ['Z_Z2', 'Z_Z1', 'Z']
    
    ST = state_tomography(3)
    ST_operators = compile_state_tomography(ST, qubits, m_operator)
    print(ST_operators)
    # e = expectation_value_calculator(ds,Witness_operators, qubits, m_operator)
    # e.add_readout_properties('ZZII', (.97, .98)) 
    # e.add_readout_properties('ZIII', (.97, .98))  
    # e.add_readout_properties('IZII', (.97, .98))  
    # e.add_readout_properties('IIZI', (.97, .98))  
    # e.add_readout_properties('IIIZ', (.97, .98))  
    # expectation_values = e.calc_expectation_values(iteration=0)

    # Witness_value = calculate_witness(W, expectation_values)

    out = decompose_measurement_operator('ZZI', [1,2,3], ['Z_Z2', 'Z_Z1', 'Z'])
    print(out)