import numpy as np

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





