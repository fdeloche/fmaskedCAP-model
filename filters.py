import numpy as np
import torch

from scipy.special import comb


##### Auditory filters #####


class GaussianFilter:
	'''
	class for Gaussian filters. Main methods are just a recall of some formulas. 
	The model mainly uses the class method 'int_standard_sq'
	'''
	def __init__(self, sig_f, f_c):
		self.sig_f=sig_f
		self.f_c=f_c

	def __call__(self, f):
		'''
		Args:
			f: float or numpy array of frequency values
		Returns:
			The frequency response at f (amplitude spectrum).  
			'''
		return 1/np.sqrt(np.sqrt(2*np.pi)*self.sig_f)*torch.exp(-1/4*(f-self.f_c)**2/self.sig_f**2)

	@classmethod
	def givenQ10(cls, f_c, Q10):
		'''
		Alternative constructor with f and Q10 values  
		'''
		BW10=f_c/Q10
		sig_f=BW10/(2*np.sqrt(2*np.log(10)))
		return cls(sig_f, f_c)

	def BW10(self):
		'''
		Returns:
			The 10dB bandwidth
		'''
		return 2*np.sqrt(2*np.log(10))*self.sig_f #2 x 2.14 sig_f

	@classmethod
	def int_standard_sq(cls, x):
		'''integrates frequency amplitude squared x=f/bw_10'''
		cte=2*np.sqrt(2*np.log(10)) #sig_f=bw10/(2*2.14...)
		return 1/2+1/2*torch.erf(x*cte/np.sqrt(2))

class GammatoneFilter:
	'''
	class for Gammatone filters. Main methods are just a recall of some formulas. 
	The model mainly uses the class method 'int_standard_sq'
	'''

	def __init__(self, k, tau, f_c):
		self.k=k
		self.tau=tau
		self.f_c=f_c

	def __call__(self, f):
		omega=2*np.pi*f
		k=self.k
		tau=self.tau
		res_sq=1/comb(2*k-2, k-1)*np.power(2, 2*k-1)*tau*torch.pow( 1+tau**2*omega**2 , -k)
		return torch.sqrt(res_sq)

	def BW10(self):
		return 1/(self.tau*np.pi)*np.sqrt(10**(1/self.k)-1)


	@classmethod
	def int_standard_sq(cls, k, x):
		'''integrates frequency amplitude squared x=f/bw_10'''

		def prim_cos_2k(k, x):
			res=0
			li=[]
			for l in range(k):
				coeff=2*comb(2*k, l)*1/(2*k-2*l)
				res+=coeff*torch.sin((2*k-2*l)*x)
			res+=comb(2*k, k)*x
			res/=2**(2*k)
			return res

		tau=1/(np.pi)*np.sqrt(10**(1/k)-1)  #(*1/bw_10)
		return 0.5+1/comb(2*k-2, k-1)*np.power(2, 2*k-2)/np.pi*prim_cos_2k(k-1, torch.arctan(2*np.pi*tau*x))

