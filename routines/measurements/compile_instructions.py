from routines.measurements.instruction_set import instruction_mgr

import numpy as np

class compile_measurements:
	def __init__(self, measurement_operators, qubits_used = [1,2,3,4,5,6], measurement_operator = ['Z_Z2', 'Z_Z1', 'Z', 'Z', 'Z_Z6', 'Z_Z5']):
		'''
		translates measurement operators to the operators available in the experiment.
		Args :
			measurement_operators (measurment_operator) : list of operators that should be measured.
			qubits_used (list) : list with the qubit number to which the measurement operators correspond
			measurement_operator (list<str>) : list of measurement operators per qubit -- available for measurement on the sample.
		'''
		self.measurement_operators = measurement_operators
		self.n_qubits = len(qubits_used)
		self.qubits = qubits_used
		self.m_op = measurement_operator

	def compile(self, name):
		'''
		compile in minimal list of measurements to use.
		'''
		return self.__compile_measurement_circuit(name)

	def __compile_measurement_circuit(self, name):
		instruction_list = [[]]*len(self.measurement_operators)

		# Convert measurement basises
		for idx, operator in enumerate(self.measurement_operators):
			op_name = np.asarray(list(operator.name))
			Y_gates = np.where(op_name=='X')[0]
			X_gates = np.where(op_name=='Y')[0]
			instr_list = []
			for j in Y_gates:
				instr_list.append(f'q{self.qubits[j]}_mY90')
			for j in X_gates:
				instr_list.append(f'q{self.qubits[j]}_X90')
			instruction_list[idx] = instr_list

		# add CNOT gates when ZZ self.measurement_operators needs to be converted into IZ/ZI
		for idx, operator in enumerate(self.measurement_operators):
			op_name = np.asarray(list(operator.name))
			m_op = np.where(op_name!='I')[0]

			for m in m_op:
				if self.m_op[m] != 'Z':
					_, correlator = self.m_op[m].split('_')
					correlator = int(''.join(j for j in correlator if j.isdigit()))
					if self.qubits.index(correlator) not in m_op:
						pair =[correlator, self.qubits[m]].sort()
						if correlator > self.qubits[m]:
							instruction_list[idx].append(f'q{self.qubits[m]}{correlator}_CNOT12')
						else:
							instruction_list[idx].append(f'q{correlator}{self.qubits[m]}_CNOT21')

		# reduce the instuction list to only the needed instrcutions
		reduced_instruction_list = instruction_mgr(name)
		
		for idx, operator in enumerate(self.measurement_operators):
			reduced_instruction_list.add(instruction_list[idx], operator)

		return reduced_instruction_list