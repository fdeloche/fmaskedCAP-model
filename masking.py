import numpy as np 



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



