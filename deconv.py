import numpy as np 

def deconv_grad(EPs, u_fft, CAPs_fft):
	'''
	Args:
		EPs: matrix of excitation patterns (in time domain)
		u_fft: unitary response (rfft)
		CAPs_fft: matrix of CAP signals (rfft) corresponding to EPs
	Returns:
		Gradient of deconvolution for EPs
	'''
	EPs_fft = np.fft.rfft(EPs, axis=1)
	grad_fft=-2*(CAPs_fft-EPs_fft*u_fft)*u_fft
	return np.fft.irfft(grad_fft, axis=1)



def deconv_newton_step(EPs, u_fft, CAPs_fft, eps=1e-6):
	'''
	Note: EPs and u_fft can be exchanged (u_mat, EPs_fft)
	Args:
		EPs: matrix of excitation patterns (in time domain)
		u_fft: unitary response (rfft)
		eps: epsilon, not to divide by zero
		CAPs_fft: matrix of CAP signals (rfft) corresponding to EPs
	Returns:
		Gradient-like term cooresponding to one step of Newton algorithm, for EPs
	'''
	EPs_fft = np.fft.rfft(EPs, axis=1)
	dEP_fft=-(CAPs_fft/(u_fft+eps)-EPs_fft)
	return np.fft.irfft(dEP_fft, axis=1)