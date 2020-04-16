import numpy as np 

from filters import *

###########  MASKING DEGREE FUNC. ##############


class MaskingDegreeFunction:
	'''Parent class for functions relating masking degree with intensity (dB).  
	It implements a linear function (on a bounded interval) by default, methods should be overridden by children.'''

	def __init__(self, a, b, max_masking=1.):
		'''
		Args:
			a: left bound of interval (dB)
			b: right bound of interval
			max_masking: maximal degree of masking (reached at b, between 0 and 1)
		'''
		self.a=a
		self.b=b
		self.max_masking=max_masking
		self.slope=max_masking/(b-a)

	def func(self, I):
		'''
		Args:
			I (float or numpy array) : Intensity
		Returns:
			float or numpy array: Masking degree (between 0 and 1)
		'''
		return (I>self.a)*(I<self.b)*self.slope*(I-self.a) + (I>= self.b)*self.max_masking

	#alias
	md=func


class SigmoidMaskingDegreeFunction(MaskingDegreeFunction):
	'''Implements the sigmoid function
	f(I) = m/(1+exp(-a(I-mu) ))
	'''
	def __init__(self, mu, a, m=1):
		'''
		Args:
			mu: point where 50% of max masking is reached
			a: shape parameter, 2x slope at I0 (considering that max masking is 100%)
			m: maximum masking
		'''
		self.mu=mu
		self.a=a
		self.max_masking=self.m=m

	def func(self, I):
		I=np.maximum(I, -20) #HACK: avoid overflow in exp
		return self.m/(1+np.exp(-self.a*(I-self.mu)))

	md=func




################ MASKING PATTERNS ####################

class MaskingPattern:
	'''Parent class for masking patterns.  
	By default, no masking.  
	Methods should be overriden by children'''

	def __init__(self):
		pass

	def M(self, f):
		'''
		Returns:
			Degree of masking as a function of frequency
		'''
		return np.zeros_like(f)

	def __repr__(self):
		return 'no masking'

class ToneSingleFilterMaskingPattern(MaskingPattern):
	'''
	Masking pattern associated with a tone masker and a model of a filter bank made of a single auditory filter model (constant bandwidth).
	'''

	def __init__(self, A, f_0, filt, mdFunc):
		'''
		Args:
			A: amplitude of tone (in dB)
			f_0: frequency of tone
			filt: AuditoryFilter object
			mdFunc: MaskingDegreeFunction object
		'''
		self.A = A
		self.f_0=f_0
		self.filt=filt
		self.mdFunc=mdFunc

	def M(self, f):
		I=20*np.log10(self.A)+self.filt.g_dB(f-self.f_0)
		return self.mdFunc.md(I)

	def __repr__(self):
		st=f"Tone masker (1-filter bank model)\n"
		st+=f"f_c={self.f_0/1e3:.2f} kHz; Amplitude:{20*np.log10(self.A):.1f} dB"
		st+=f"\nfilter model: {self.filt}"
		return st


