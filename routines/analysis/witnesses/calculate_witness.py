def calculate_witness(witness_operators, expectation_values):
    W  = 0
    for operator in witness_operators:
        W += operator.prob*expectation_values[operator.name]
    return W

if __name__ == '__main__':
	from routines.analysis.get_expectation_value import expectation_value_calculator

	pass