from dataclasses import dataclass
import numpy as np

@dataclass
class reduced_instruction:
	m_id		  : int
	instruction : list
	operators	  : list

	@property
	def hash(self):
		return str(self.instruction)

class instruction_mgr:
	def __init__(self, name):
		self.name = name 
		self.instructions = []

	def add(self, instruction_list, operator):
		instr = reduced_instruction(len(self.instructions), instruction_list, [operator])
		for i in self.instructions:
			if i.hash == instr.hash and operator is not None:
				i.operators += instr.operators
				return
		self.instructions.append(instr)

	def flatten(self):
		new = instruction_mgr(self.name)
		for instruction in self:
			for op in instruction.operators:
				new.instructions.append(reduced_instruction(instruction.m_id, None, [op]))
		return new 
		
	def __len__(self):
		return len(self.instructions)

	def __iter__(self):
		self.n = 0
		return self

	def __next__(self):
		if self.n >= len(self):
			raise StopIteration
		self.n += 1
		return self[self.n-1]
	
	def __getitem__(self, idx):
		return self.instructions[idx]

	def __repr__(self):
		represenation = 'Instructions present in this instruction set :: \n'
		represenation += '| {:<10}| {:<70}| {:<70}|\n\n'.format('id', 'operators', 'instructions')
		for i in self.instructions:
			represenation += f'| {i.m_id:<10}|'
			op_str = ''
			for op in i.operators:
				if op is not None:
					op_str += f'{op.name}, '
			represenation += f' {op_str[:-2]:<70}| {str(i.instruction):<70}|\n'
		return represenation