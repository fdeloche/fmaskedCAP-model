import torch
import copy
import numpy as np

class PowerLawLatencies:
	'''
	Links frequencies and latencies (power law model).

	log(f)= log(A) + alpha log (t-t0) 
	Attributes:
		A
		alpha
		t0
	'''

	def __init__(self, A, alpha, t0=0., requires_grad=False, name=""):
		self.A=torch.tensor(A, requires_grad=requires_grad)
		self.alpha=torch.tensor(alpha, requires_grad=requires_grad)
		self.t0=torch.tensor(t0, requires_grad=requires_grad)
		self.name=name

	@classmethod
	def fromPts(cls, t1, f1, t2, f2, t0=0, name=""):
		det = np.log(t2-t0)-np.log(t1-t0)
		alpha=(np.log(f2)-np.log(f1))/det
		beta= (np.log(t2-t0)*np.log(f1)-np.log(t1-t0)*np.log(f2))/det
		return cls(np.exp(beta), alpha, t0, name=name)

	@classmethod
	def dilate(cls, lat, a, f_0):
		'''
		Local dilatation with fixed point at f_0
		Class method.
		Args:
			lat: PowerLawLatencies object
			a : dilatation factor
			f_0: frequency for fixed point
		Returns:
			A new PowerLawLatencies object
		'''
		with torch.no_grad():
			t_f0=lat.t_from_f(f_0)
			alpha=lat.alpha.clone()/a
			A=f_0/torch.pow(t_f0-lat.t0, alpha)
		return cls(A, alpha, t0=lat.t0)

	@classmethod
	def shift(cls, lat, t0, reinitShift=False):
		'''
		Class method.
		Args:
			lat: PowerLawLatencies object
			t0 (float) : shift in s (warning: it adds to the initial value if it applies, unless reinitShift is True) 
		Returns:
			A new PowerLawLatencies object, lat shifted by t0 
		'''
		with torch.no_grad():
			alpha=lat.alpha.clone()
			A=lat.A.clone()
			t0=(1-reinitShift)*lat.t0+t0
		return cls(A, alpha, t0=t0)
		
	def f_from_t(self, t):
		return self.A*torch.pow(t-self.t0, self.alpha)

	def t_from_f(self, f):
		return self.t0 + torch.pow(f/self.A, 1/self.alpha)

	def __call__(self, f):
		return self.t_from_f(f)

	def __repr__(self):
		return f"PowerLawLatencies obj. {self.name} \n (A={self.A.numpy():.2e}, alpha={self.alpha.numpy():.2e}, t0={self.t0.numpy():.2e})"




