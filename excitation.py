import torch
import tuning
import masking
import numpy as np

#from scipy.stats import gamma

def get_sq_masking_excitation_patterns(f, bw10Func, n_conditions, n_bands, amp_list, f_low_list, f_high_list, filter_model='gaussian'):
	'''
	Args:
		f: Tensor of frequencies at which the excitation is computed 
		bw10Func: function (or callable) to get BW10, see tuning
		n_conditions: number of masking conditions
		n_bands: (max) number of bands for the noise masking conditions
		amp_list: list of Tensors (length list: n_bands, dim Tensors: nb_conditions) amp of each band 
		f_low_list: list of Tensors (length list: n_bands, dim Tensors: nb_conditions) f_cut low for each band
		f_high_list: list of Tensors (length list: n_bands, dim Tensors: nb_conditions) f_cut high
		filter_model: 'gaussian' (default), only available for now  #TODO gammatone filters
	Returns:
		squared excitation patterns associated to the masking conditions (tensor shape (n_conditions, n_freq))
	'''

	bw10=bw10Func(f)
	bw10_inv=1/bw10
	if filter_model=='gaussian': 

		cte=2*np.sqrt(2*np.log(10)) #sig_f=bw10/(2*2.14...)
		def F(x):
			return 1/2+1/2*torch.erf(x*cte/np.sqrt(2))   

	exc=torch.zeros(n_conditions, f.shape[0])

	assert len(amp_list)==len(f_low_list)==len(f_high_list)==n_bands, 'band param lists must be of length n_bands'

	for amp, f_low, f_high in zip(amp_list, f_low_list, f_high_list):
		b=(torch.unsqueeze(f_high, 1) - torch.unsqueeze(f, 0))*bw10_inv
		a=(torch.unsqueeze(f_low, 1) - torch.unsqueeze(f, 0))*bw10_inv
		exc+= torch.unsqueeze(amp**2, 1)*(F(b)-F(a))
	return exc


def get_sq_masking_excitation_patterns_maskCond(f, bw10Func, maskCond, filter_model='gaussian'):
	'''overloading function for get_sq_masking_excitation_pattern'''
	return get_sq_masking_excitation_patterns(f, bw10Func, maskCond.n_conditions, maskCond.n_bands, maskCond.amp_list, maskCond.f_low_list, maskCond.f_high_list, filter_model=filter_model)


class ExcitationPatterns:

	def __init__(self, t, E0_maskable, E0_nonmaskable=None, requires_grad=False):
		'''
		Args:
			t: time vector (torch tensor)
			E0_maskable: raw excitation pattern (numpy array or torch tensor), maskable part

			E0_nonmaskable (optional): raw excitation pattern (numpy array or torch tensor), fixed part
			requires_grad: if E0 tensors requires gradient mdFunc
		'''
		self.t=t
		if torch.is_tensor(E0_maskable):
			self.E0_maskable = E0_maskable.clone().detach().requires_grad_(requires_grad=requires_grad)
		else:
			self.E0_maskable=torch.tensor(E0_maskable, requires_grad=requires_grad)
		if E0_nonmaskable is None:
			self.E0_nonmaskable=torch.zeros_like(self.E0_maskable)
		else:		
			if torch.is_tensor(E0_nonmaskable):
				self.E0_nonmaskable = E0_nonmaskable.clone().detach().requires_grad_(requires_grad=requires_grad)
			else:
				self.E0_nonmaskable=torch.tensor(E0_nonmaskable, requires_grad=requires_grad)
		self.masked=False

	def set_masking_model(self, latencies, bw10Func, maskCond, maskingIOFunc, filter_model='gaussian'):
		'''
		Args:
			maskCond: MaskingConditions object
			maskingIOFunc: e.g. SigmoidMaskingDegreeFunction object
		'''
		self.masked=True
		self.latencies=latencies
		self.bw10Func=bw10Func
		self.maskingConditions=maskCond
		self.maskingIOFunc=maskingIOFunc
		self.filter_model=filter_model


	def get_tensors(self, eps=1e-6):
		'''
		Returns:
			a tuple (maskingAmounts, excitation_patterns) of tensors  (of shape (n_conditions, n_freq))
		'''
		
		if self.masked:
			f=self.latencies.f_from_t(self.t)
			sq_masking_exc_patterns=get_sq_masking_excitation_patterns_maskCond(f, self.bw10Func, self.maskingConditions, filter_model=self.filter_model)
		
			I=10*torch.log10(sq_masking_exc_patterns+eps)
			maskingAmount=self.maskingIOFunc(I)
			return maskingAmount, torch.unsqueeze(self.E0_nonmaskable, 0)+torch.unsqueeze(self.E0_maskable, 0)*(1-maskingAmount)
		else:
			return None, torch.unsqueeze(self.E0_nonmaskable+self.E0_maskable, 0)  #init with raw excitation pattern
		
	def get_tensor(self, eps=1e-6):
		'''
		Returns:
			a tensor of shape (n_conditions, n_freq) representing the excitation patterns
		'''
		maskingAmounts, excitation_patterns = self.get_tensors(eps=eps)
		return excitation_patterns

	@classmethod
	def GammaExcitation(cls, t, C, alpha, beta, loc, C_nm=0, alpha_nm=1, beta_nm=1, loc_nm=0):
		'''
		Creates an excitation from a gamma law distribution. Max amp: C 
		_nm params are for the nonmaskable part (optional)
		Gamma CDF: G(t) = beta/Gamma(a) (beta (t-loc) )**(alpha-1)*exp(-beta (t-loc)) 
		Normalized function: G(t)/G( (alpha-1)/beta - loc)= (beta*(t-loc)/(alpha-1))**(alpha-1) * exp( alpha - 1 - beta (t-loc))
		'''
		assert beta>0, 'beta must be a positive number'
		#E0_maskable=gamma.pdf(t, alpha, loc, 1/beta)
		#E0_maskable*=C/gamma.pdf( (alpha-1)/beta, alpha, loc, 1/beta)
		tt=(t-loc)*((t-loc)>0)+loc
		E0_maskable=C*(beta*(tt-loc)/(alpha-1))**(alpha-1)*np.exp(alpha - 1 - beta*(tt-loc))

		if C_nm !=0:
			assert beta_nm>0, 'beta must be a positive number'
			E0_nonmaskable=C_nm*(beta_nm*(tt-loc_nm)/(alpha_nm-1))**(alpha_nm-1) *np.exp(alpha_nm - 1 - beta_nm*(tt-loc_nm))
		else:
			E0_nonmaskable=None

		return cls(t, E0_maskable, E0_nonmaskable=E0_nonmaskable)

	@classmethod
	def copyRaw(cls, E, requires_grad=False):
		'''creates a raw excitation pattern by making a copy from another ExcitationPatterns object'''
		return cls(E.t, E.E0_maskable, E0_nonmaskable=E.E0_nonmaskable, requires_grad=requires_grad)
