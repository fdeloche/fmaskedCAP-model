import torch


class constant_BW10:
	def __init__(self, BW10_0, requires_grad=False):
		'''
		Args:
			BW10_0: initial bandwitdh (Hz) at -10dB
			requires_grad: forwards argument to pytorch
		''' 
		self.BW_10=torch.tensor(BW10_0, requires_grad=requires_grad)

	def __call__(self, f):
		return self.BW_10


