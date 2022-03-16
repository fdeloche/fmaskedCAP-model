import torch
import numpy as np

import tuning
from excitation import get_sq_masking_excitation_patterns_maskCond


class LogLinearSuppression:
	def __init__(self, a, I0_supp, requires_grad=False):
		'''
		Args:
			a: suppression amount dB/dB
		'''
		self.a=torch.tensor(a, requires_grad=requires_grad)
		self.I0_supp=torch.tensor(I0_supp, requires_grad=requires_grad)

	def set_I0_supp(self, I0_supp):
		self.I0_supp.data=I0_supp

	def __call__(self, I_supp):
		return self.a*(I_supp-self.I0_supp)


class SoftPlusSuppression:
	def __init__(self, a, b, I0_supp, requires_grad=False):
		self.a=torch.tensor(a, requires_grad=requires_grad)
		self.b=torch.tensor(a, requires_grad=requires_grad)
		self.I0_supp=torch.tensor(I0_supp, requires_grad=requires_grad)

	def __call__(self, I_supp):
		return self.a*torch.log(1+torch.exp(self.b*(I_supp-self.I0_supp )))


class SuppressionAmount:
	'''util class to compute the amount of suppression as a function of frequency'''
	def __init__(self, suppFunc, suppBW10Func, freq_factor, filter_model='gaussian'):
		'''
		Args:
			suppFunc: suppression function (suppression amount computed as a function of I_supp)
			suppBW10: suppresion tuning
			freq_factor: computes I_supp based on freq_factor*f
			filter_model: forwarded to get_sq_masking_excitation_patterns
		'''
		self.suppFunc=suppFunc
		self.suppBW10Func=suppBW10Func
		self.freq_factor=freq_factor
		self.filter_model=filter_model

	def __call__(self, f, maskCond, eps=1e-6):
		exc_sq=get_sq_masking_excitation_patterns_maskCond(f*self.freq_factor, self.suppBW10Func, 
			maskCond, filter_model=self.filter_model)
		I_supp=10*torch.log10(exc_sq+eps)
		return self.suppFunc(I_supp)

