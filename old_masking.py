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

NoMaskingPattern = MaskingPattern

class MaskingCondition:
	'''Parent class for masking conditions. By default, no masking.
	Object that generates masking patterns'''


	def __init__(self):
		pass

	def pattern(self, filt, mdFunc):
		'''
		Returns:
			a NoMaskingPattern
		'''
		return NoMaskingPattern()

	def __repr__(self):
		return 'no masking'

#alias
class NoMaskingCondition(MaskingCondition):
	pass

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
		I=20*np.log10(self.A)+self.filt.g_dB(self.f_0-f)
		return self.mdFunc.md(I)

	def __repr__(self):
		st=f"Tone masker (1-filter bank model)\n"
		st+=f"f_c={self.f_0/1e3:.2f} kHz; Amplitude:{20*np.log10(self.A):.1f} dB"
		st+=f"\nfilter model: {self.filt}"
		return st

class ToneSingleFilterMaskingCondition(MaskingCondition):
	def __init__(self, A, f_0):
		'''
		Args:
			A: amplitude of tone (in dB)
			f_0: frequency of tone
		'''
		self.A=A
		self.f_0=f_0

	def pattern(self, filt, mdFunc):
		'''
		Args:
			filt: AuditoryFilter object
			mdFunc: MaskingDegreeFunction object
		Returns:
			a ToneSingleFilterMaskingPattern corresponding to tone and filter model/mdFunc
		'''
		return ToneSingleFilterMaskingPattern(self.A, self.f_0, filt, mdFunc)

	def __repr__(self):
		st=f"Tone masker (f_c={self.f_0/1e3:.2f} kHz; Amplitude:{20*np.log10(self.A):.1f} dB)"
		return st

class HighPassNoiseSingleFilterMaskingPattern(MaskingPattern):
	'''
	Masking pattern associated with a high-passed noise masker and a model of a filter bank made of a single auditory filter model (constant bandwidth).
	'''

	def __init__(self, IHz, f_cut, filt, mdFunc):
		'''
		Args:
			IHz: Noise power spectrum distribution weight (in dB/Hz)
			f_cut: cut-off frequency for high passed noise
			filt: AuditoryFilter object
			mdFunc: MaskingDegreeFunction object
		'''
		self.IHz = IHz
		self.f_cut=f_cut
		self.filt=filt
		self.mdFunc=mdFunc


	def M(self, f):
		I=self.IHz+self.filt.right_sq_int_dB(self.f_cut-f)
		return self.mdFunc.md(I)

	def __repr__(self):
		st=f"High-pass noise masker (1-filter bank model)\n"
		st+=f"f_cut={self.f_cut/1e3:.2f} kHz; PSD weight:{self.IHz:.2f} dB"
		st+=f"\nfilter model: {self.filt}"
		return st


class HighPassNoiseSingleFilterMaskingCondition(MaskingCondition):
	def __init__(self, IHz, f_cut):
		'''
		Args:
			IHz: Noise power spectrum distribution weight (in dB/Hz)
			f_cut: cut-off frequency for high passed noise
		'''
		self.IHz = IHz
		self.f_cut=f_cut

	def pattern(self, filt, mdFunc):
		'''
		Args:
			filt: AuditoryFilter object
			mdFunc: MaskingDegreeFunction object
		Returns:
			a HighPassNoiseSingleFilterMaskingPattern corresponding to noise and filter model/mdFunc
		'''
		return HighPassNoiseSingleFilterMaskingPattern(self.IHz, self.f_cut, filt, mdFunc)

	def __repr__(self):
		st=f"High-pass noise masker (f_cut={self.f_cut/1e3:.2f} kHz; PSD weight:{self.IHz:.2f} dB)"
		return st
