import numpy as np
import csv

from latencies import *
from excitation import *

import copy
import functools



class ConvolutionCAPSimulatorSingleFilterModel:
	'''
	Simulates CAPs based on a raw excitation pattern, a set of masking patterns, and a unitary response.

	Works with arrays (not ExcitationPattern, MaskingCondition objects, etc., except for initalization). 
	Then, the array t should be considered invariant throughout simulations.

	The masking patterns are based on a list of masking conditions and a single filter model for auditory filters (constant bandwidth)

	Attributes:
		E0 (array): raw excitation pattern (without masking). Note: it is advised to have NoMaskingCondition in list of masking conditions at init.
		t (numpy array): time
		E_init : E0 at initialization of object (ExcitationPattern object)
		m: number of masking patterns
		maskingPatterns: numpy arrays of 'masking patterns' where each row corresponds to a masking condition
		u: unitary response (array)
	'''

	def __init__(self, lat, filt, E_init, mdFunc, maskingConditions, t=None, ur=None):
		'''
		Optional args:
			lat: Latencies object
			filt: filter model
			E_init: ExcitationPattern object, for initialization of excitation pattern (defined hereafter as an array)
			t: time for CAP simulations, if None, initialized with np.linspace(5e-4, 10e-3, num=500)
			ur (UnitaryResponse object or array): if None intialized with a zero array except 1 at pos 0
			mdFunc: maskingDegreeFunction
			maskingConditions: a list of maskingConditions to create the masking patterns.
		'''

		if t is None:
			t=self.default_t_array()
		self.t=t

		self.latencies=lat

		self.E_init=E_init #initial excitation pattern
		self.E0=E_init.E(t) #array

		self.filt=filt
		self.mdFunc=mdFunc
		self.maskingConditions_init=maskingConditions

		#create masking patterns
		maskingPatterns=np.zeros((len(maskingConditions), len(t)))
		self.m=len(maskingPatterns)
		f = self.latencies.f_from_t(t)
		for i, mc in enumerate(maskingConditions):
			MPat=mc.pattern(filt, mdFunc)
			maskingPatterns[i]=MPat.M(f)
		self.maskingPatterns=maskingPatterns

		#unitary response
		self.ur_init=ur
		if ur is None:
			self.u=np.zeros_like(t)
			self.u[0]=1
		else:
			if isinstance(ur, URfromCsv):
				self.u = ur.u(t)
			else: #np array
				assert len(ur)==len(t), 'u and t must be of same size'
				self.u = ur

	@classmethod
	def fromNewExcitationPattern(cls, capSimulator, E0):
		'''
		Args:
			E0 (np array) : raw excitation pattern 
		Returns:
			a (shallow) copy of capSimulator with new excitation pattern E0
		'''
		capSimulator2=copy.copy(capSimulator)
		capSimulator2.set_E0(E0)
		return capSimulator2

	@classmethod
	def default_t_array(cls):
		return np.linspace(5e-4, 10e-3, num=500)

	def getExcitationPatterns(self):
		'''
		computes excitation patterns with E=E0*(1-M)
		Returns:
			numpy array: excitations patterns as a matrix (similar to maskingPatterns)
		'''
		return self.E0*(1-self.maskingPatterns)

	getEPs=getExcitationPatterns #alias


	def simulCAPs(self):
		'''
		Simulates CAPs by convolution of excitation patterns and u
		Returns:
			numpy array: simulations of CAP as a matrix (similar to maskingPatterns)
		'''
		EPs=self.getEPs()
		m, T = np.shape(EPs)
		CAPs=np.zeros((m,T))
		for i in range(m):
			CAPs[i]=np.convolve(EPs[i], self.u, 'full')[0:T]
		return CAPs

	def get_projector(self):
		'''
		Returns:
			A function that projects a matrix of excitation patterns in the linear subspace
			 ((1-M_0) E_0, ... (1-M_(m-1)) E_0) parametrized by E0. (linear regression)
			Argument of the resulting function is E_mat matrix of excitation patterns.
		'''
		def proj(maskingPatterns, E_mat):
			cross_prod=np.average((1-maskingPatterns)*E_mat, axis=0)
			mean_square=np.average((1-maskingPatterns)**2, axis=0)
			E0=cross_prod/mean_square
			return (1-maskingPatterns)*E0

		return functools.partial(proj, self.maskingPatterns)

	def set_E0(self, E0):
		self.E0=E0

	def set_u(self, u):
		self.u=u




########### 'REALISTIC' SIMULATIONS ###########


class URfromCsv:

	def __init__(self,filename, name='', shifted=False):

		'''
		Args:
			filename (str) : csv filename. 2 fields: time (in ms), amplitude
			shifted (boolean) : if True, time array begins at 0
		'''
		with open(filename, 'r') as f:
			csv_reader=csv.DictReader(f, delimiter=',')
			t=[]
			u=[]

			#
			for column in csv_reader.fieldnames:
				if 'time' in column.lower():
					tCol = column 
				if 'amplitude' in column.lower():
					uCol=column

			for row in csv_reader:
				t.append(float(row[tCol])*1e-3)
				u.append(float(row[uCol]))



			t,u= np.array(t), np.array(u)

			ind=np.argsort(t)
			self._t, self._u = t[ind], u[ind]
			if shifted:
				self._t-=t[0]
		self._func = lambda x: 1
		self.name=name

	@classmethod
	def modify(cls, ur, func, name=None):
		'''
		Returns a new UR based on ur with a function (of t). u will be initial ur multiplied by func. Based on shallow copy
		''' 
		ur2=copy.copy(ur)
		ur2._func=func
		if name is not None:
			ur2.name=name
		else:
			ur2.name = ur2.name + " modified"
		return  ur2


	def u(self, t):
		'''
		Args:
			t (float or numpy array)
		Returns:
			float or numpy array: UR for t
		'''
		return np.interp(t, self._t, self._u, left=0., right=0.)*self._func(t)

URWang1979 = URfromCsv('./UR/Wang1979Fig14.csv', name='averaged UR (Wang 1979, Fig14)')
URWang1979m = URfromCsv.modify(URWang1979, lambda t:1+4*np.exp(-1/2*(t+0.2e-4)**2/1e-4**2)) #produces more realistic CAP

URWang1979shifted = URfromCsv('./UR/Wang1979Fig14.csv', name='averaged UR (Wang 1979, Fig14)', shifted=True)

Eggermont1976clickLatencies80dB=PowerLawLatencies.fromPts(5.3e-3, 1e3, 2e-3, 5e3, name="Eggermont 1976 click 80dB")

def simulCAP2convolGaussianKernel(E, t=None, ur=URWang1979m, sig=5e-4, 
	secondPeak=0., secondPeak_sig=3e-4, secondPeak_dt=1.5e-3):
	'''
	Args:
		E: ExcitationPattern
		t (array-like): time, if None, init with np.linspace(5e-4, 10e-3, num=500)
		ur: Unitary response object
		sig: std of Gaussian kernel 'blur' (in s)
		secondPeak (float): float between 0 and 1 (typically), amplitude ratio second peak/first speak. 0: no second peak
		secondPeak_sig (float): std for addition blur on second peak
		secondPeak_dt (float): time between first and second peak
	Returns:
		array-like: a simulation of the CAP
	'''

	
	from scipy.ndimage  import gaussian_filter1d
	if t is None:
		t=np.linspace(5e-4, 10e-3, num=500)
	#find secondPeak_dt in number of indices
	secPind=int(np.sum((t-t[0])<secondPeak_dt))

	#second peak
	ExP= E.E(t)
	Exp_pad=np.pad(ExP, (0, secPind))
	ExP2= np.roll(Exp_pad, secPind)
	ExP2=ExP2[:len(ExP)]
	#additional blur
	secondPeak_sig2=secondPeak_sig/(t[1]-t[0])
	ExP2=gaussian_filter1d(ExP2, sigma=secondPeak_sig2)
	ExP2*=secondPeak
	ExP+=ExP2

	#find indices between 0 and 2 ms
	ind=t<2e-3
	t2=t[ind]
	u=ur.u(t2-1e-3) #NB shift +1 ms
	v=np.convolve(ExP, u, 'full')

	#find first indice > 2 ms
	for i in range(len(t)):
		if t[i]>1e-3:
			break
	v=v[i:i+len(t)] #shift again

	#model PSTH
	sig2=sig/(t[1]-t[0])
	v=gaussian_filter1d(v, sigma=sig2)
	return v
