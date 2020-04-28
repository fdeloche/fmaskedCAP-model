import copy
import numpy as np

class PowerLawLatencies:
	'''
	Relates frequency with latencies (power law model).

	log(f)= log(A) + alpha log (t-t0) 
	Attributes:
		A
		alpha
		t0
	'''

	def __init__(self, A, alpha, t0=0., name=""):
		self.A=A
		self.alpha=alpha
		self.t0=t0
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
		dLat=copy.copy(lat)
		t_f0=lat.t_from_f(f_0)
		dLat.alpha/=a
		dLat.A=f_0/np.power(t_f0-dLat.t0, dLat.alpha)
		return dLat

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
		sLat=copy.copy(lat)
		t0=(1-reinitShift)*lat.t0+t0
		sLat.t0=t0
		return sLat

	def f_from_t(self, t):
		return self.A*np.power(t-self.t0, self.alpha)

	def t_from_f(self, f):
		return self.t0 + np.power(f/self.A, 1/self.alpha)

	def __repr__(self):
		return f"PowerLawLatencies obj. {self.name} \n (A={self.A:.2e}, alpha={self.alpha:.2e}, t0={self.t0:.2e})"




