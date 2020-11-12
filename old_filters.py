import numpy as np
from scipy.integrate import cumtrapz

class AuditoryFilter:
	'''Parent class for auditory filters. It implements a rectangular filter but its methods should be overridden by children.'''

	def __init__(self, BW):
		'''
		SHOULD BE OVERRIDEN
		Args:
			BW (float): Bandwidth or the rectangular filter
		'''
		self.BW=BW
		self.name='Rectangular filter (default class)'

	def g(self, f):
		'''
		SHOULD BE OVERRIDEN
		Args:
			f: float or numpy array of frequency values
		Returns:
			The frequency response at f (amplitude spectrum).  
			The filter is assumed to be centered around 0 and (L2-) normalized to 1.'''
		return (np.abs(f)<self.BW/2)*1./np.sqrt(self.BW)

	def g_dB(self, f, eps=1e-6):
		'''
		Args:
			f: float or numpy array of frequency values
		Returns:
			Power spectrum in dB
		'''
		return 20*np.log10(self.g(f)+eps)

	def _compute_right_sq_int(self, fmax=None, num=5000):
		'''Private method. Computes and stores the values of integral from +inf of squared filter in a array
		Args:
			f_max: integral will be computed between -f_max and f_max. If none, f_max is chosen as 3 BW10.
			num: num of points to compute the integral values'''
		if fmax is None:
			fmax=3*self.BW10()
		f=np.linspace(fmax, -fmax, num)
		self._right_sq_int=-cumtrapz(np.abs(self.g(f))**2, f, initial=0)[::-1]
		self._right_sq_int_f=f[::-1]

	def right_sq_int(self, f_cut):
		'''
		Args:
			f_cut (array-like): frequencies where to compute the integral values
		Return:
			(numpy array): The integral values of squared filter fron f_c to +inf
		'''
		if not(hasattr(self, '_right_sq_int')):
			self._compute_right_sq_int() #compute integral values at regulat intervals
		return np.interp(f_cut, self._right_sq_int_f, self._right_sq_int, 1., 0.)

	def right_sq_int_dB(self, f_cut, eps=1e-12):
		'''
		Args:
			f_cut (array-like): frequencies where to compute the integral values
		Return:
			(numpy array): The integral values of squared filter fron f_c to +inf (RMS values in dB)
		'''
		return 10*np.log10(self.right_sq_int(f_cut)+eps)

	def ERB(self):
		'''
		Returns:
			The equivalent rectangular bandwidth.
		'''
		return 1./self.g(0)**2


	def BW10(self):
		'''
		SHOULD BE OVERRIDEN
		Returns:
			The 10dB bandwidth
		'''
		return 1./self.g(0)**2

	def __repr__(self):
		st=self.name
		st+='\n'
		st+=f"ERB : {self.ERB():.1f} Hz "
		st+='\n'
		st+=f"BW10 : {self.BW10():.1f} Hz"
		st+='\n'
		st+=f"Q_ERB/Q_10 ratio : {self.BW10()/self.ERB():.3f}"
		st+='\n'
		return st


# --------------------------------\


class GaussianFilter(AuditoryFilter):
	def __init__(self, sig_f):
		'''
		Args:
			sig_f: Frequency deviation (std deviation for squared filter)
		'''
		self.sig_f=sig_f
		self.name=f'Gaussian filter (sig_f: {self.sig_f:.1f}f Hz)'


	@classmethod
	def givenQ10(cls, f_c, Q10):
		'''
		Alternative constructor with f and Q10 values  
		Note that these values will just be used to compute sig_f will and f and Q will be forgetten
		'''
		BW10=f_c/Q10
		sig_f=BW10/(2*np.sqrt(2*np.log(10)))
		return cls(sig_f)

	def g(self, f):
		'''
		Args:
			f: float or numpy array of frequency values
		Returns:
			The frequency response at f (amplitude spectrum).  
			The filter is assumed to be centered around 0 and (L2-) normalized to 1.'''
		return 1/np.sqrt(np.sqrt(2*np.pi)*self.sig_f)*np.exp(-1/4*f**2/self.sig_f**2)

	def g_dB(self, f):
		return 20*( -1/4*np.log10(2*np.pi)-1/2*np.log10(self.sig_f)-1/np.log(10)*1/4*f**2/self.sig_f**2)

	def BW10(self):
		'''
		Returns:
			The 10dB bandwidth
		'''
		return 2*np.sqrt(2*np.log(10))*self.sig_f #2 x 2.14 sig_f





