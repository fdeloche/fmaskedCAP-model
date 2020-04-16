import numpy as np  
from masking import MaskingPattern
from latencies import *
import copy

class ExcitationPattern:
	'''Parent class for excitation patterns.   
	By default, implements a quadratic excitation pattern (w.r.t frequency), i.e. similar to what a model of a Gaussian filter bank would produce.  
	Methods should be overriden by children.

	Attributes:
		latencies: Latencies object (relates t to f)
		masker: MaskingPattern object (by default no masking)
	'''

	def __init__(self, lat, f_c, Q10, A_max, A_0=20):
		'''
		Args:
			lat: latencies
			f_c: center frequency
			Q10: Q10
			A_max: amplitude at f_c (in dB)
			A_0: amplitude threshold (by default 20dB)
		'''
		BW10=f_c/Q10
		self.sig_f=BW10/(2*np.sqrt(2*np.log(10)))
		self.f_c=f_c
		self.latencies = lat
		self.masker=MaskingPattern()
		self.A_0=A_0
		self.A_max=A_max

	def E0(self, t):
		'''
		Returns:
			array-like: raw excitation pattern (without masking)
		''' 
		f = self.latencies.f_from_t(t)
		return self.E0_from_f(f)

	def E0_from_f(self, f):
		'''
		SHOULD BE OVERRIDEN BY CHILDREN
		Returns:
			array-like: raw excitation pattern (without masking)
		''' 
		# cte for normalized filter: -1/4*np.log10(2*np.pi)-1/2*np.log10(self.sig_f)
		return np.maximum(self.A_max-self.A_0-20*1/np.log(10)*1/4*(f-self.f_c)**2/self.sig_f**2, 0)

	def E(self, t):
		'''
		Returns:
			array-like: excitation pattern (with masker, i.e. E0*(1-M) )
		''' 

		f = self.latencies.f_from_t(t)
		return self.E_from_f(f)

	def E_from_f(self, f):
		'''
		Returns:
			array-like: excitation pattern (with masker, i.e. E0*(1-M) )
		''' 
		return self.E0_from_f(f)*(1-self.masker.M(f))

	def changeMasker(self, mask):
		self.masker=mask

	@classmethod
	def mask(cls, excitationPattern, mask):
		'''
		Returns:
			A new ExcitationPattern object based on excitationPattern with masker mask. (Caution: based on shallow copy of excitationPattern)
		'''
		newExcitationPattern=copy.copy(excitationPattern)
		newExcitationPattern.changeMasker(mask)
		return newExcitationPattern

	@classmethod
	def rawExcitationPattern(cls, excitationPattern):
		'''
		Returns:
			A new ExcitationPattern object based on excitationPattern without any masking. (Caution: based on shallow copy of excitationPattern)
		'''
		newExcitationPattern=copy.copy(excitationPattern)
		newExcitationPattern.changeMasker(MaskingPattern())
		return newExcitationPattern

	def __repr__(self):
		return f"Excitation Pattern of quadratic type, f_0={self.f_c*1e-3:.2f} kHz\nMasker: {self.masker}"






