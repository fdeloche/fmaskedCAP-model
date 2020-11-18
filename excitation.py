import torch
import tuning
import masking
import numpy as np

def get_sq_masking_excitation_pattern(f, bw10Func, n_conditions, n_bands, amp_list, f_low_list, f_high_list, filter_model='gaussian'):
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
		squared excitation pattern associated to the masking conditions
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
		exc+= torch.unsqueeze(amp, 1)*(F(b)-F(a))
	return exc
