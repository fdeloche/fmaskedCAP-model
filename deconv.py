import numpy as np 

from scipy.ndimage  import gaussian_filter1d



def deconv_grad(EPs, u_fft, CAPs_fft, eps_ridge=0):
	'''
	Args:
		EPs: matrix of excitation patterns (in time domain)
		u_fft: unitary response (rfft)
		CAPs_fft: matrix of CAP signals (rfft) corresponding to EPs
		eps_ridge: eps of ridge regression min |ax-b|^2 + eps |x|^2
	Returns:
		Gradient of deconvolution for EPs
	'''
	EPs_fft = np.fft.rfft(EPs, axis=1)

	cross_prod=CAPs_fft*np.conjugate(u_fft)


	#grad_fft=-2*(CAPs_fft-EPs_fft*u_fft)*u_fft+2*eps_ridge*EPs_fft   #would work if arrays were real
	grad_fft=2*(-cross_prod+ (np.abs(u_fft)**2 + eps_ridge)*EPs_fft )

	return np.fft.irfft(grad_fft, axis=1)



def deconv_newton_step(EPs, u_fft, CAPs_fft, eps=1e-6, eps_ridge=0, proj_fft=None):
	'''
	Note: EPs and u_fft can be  interchanged (u_mat, EPs_fft)
	Args:
		EPs: matrix of excitation patterns (in time domain)
		u_fft: unitary response (rfft)
		CAPs_fft: matrix of CAP signals (rfft) corresponding to EPs
		eps: epsilon, in order not to divide by zero
		eps_ridge: eps of ridge regression min (ax-b)^2 + eps x^2
		proj_fft: project gradient (in freq domain) with function proj_fft, output expected to be of dim 1
	Returns:
		Gradient-like term corresponding to one step of Newton algorithm, for EPs
	'''
	EPs_fft = np.fft.rfft(EPs, axis=1)
	cross_prod=CAPs_fft*np.conjugate(u_fft)
	grad_EP=2*(-cross_prod+ (np.abs(u_fft)**2 + eps_ridge)*EPs_fft )
	dEP_fft=grad_EP/(2*np.abs(u_fft)**2+eps+2*eps_ridge)
	if proj_fft is not None:
		dEP_fft=proj_fft(dEP_fft)
		return np.fft.irfft(dEP_fft)
	return np.fft.irfft(dEP_fft, axis=1)


def blind_deconv_alternate_steps(EPs_init, u_init, CAPs, nb_alternations, nb_steps, alpha, proj_EPs, 
	proj_u, sig_d, maskingPatterns):
	'''
	Comment:not used

	Deconv for u first -> change that? 
	Low pass on E #HACK
	Args:
		nb_alternations:
		nb_steps: nb steps for each deconvolution (gradient descent)
		EPs_init: matrix of excitation patterns (initialisation)
		CAPs: CAP signals (as a matrix)
		u_init: unitary response (array, initialization)
		alpha: array of step sizes (should be of size nb_steps)
		proj_EPs: function for projection of EPs (as a matrix) after each gradient step
		proj_u: function for projection of Us (as a matrix!) after each gradient step
		sig_d: std deviation for gaussian filter on E0 (for now #HACK)
		maskingPatterns: #HACK for now
	Returns:
		EPs matrix at end of algorithm
		u (unitary response) at end of algorithm
	'''
	CAPs_fft=np.fft.rfft(CAPs, axis=1)

	EP_deconv=EPs_init
	u1 = u_init
	m=np.shape(EP_deconv)[0]

	for k in range(nb_alternations):
		EPs_fft=np.fft.rfft(EP_deconv, axis=1)
		u1_mat=np.repeat(u1[None, :], m, axis=0)
		for i in range(1, nb_steps+1):
			du=deconv_newton_step(u1_mat, EPs_fft, CAPs_fft)
			u1_mat-=alpha[i-1]*du
			#proj
			u1_mat=proj_u(u1_mat)
		u1=np.mean(u1_mat, axis=0)
		# if k==0: #first alternation
		#     u1_0=u1[:]
		u_fft=np.fft.rfft(u1)
		for i in range(1, nb_steps+1):
			dEP=deconv_newton_step(EP_deconv, u_fft, CAPs_fft)
			EP_deconv-=alpha[i-1]*dEP
			#proj
			EP_deconv=proj_EPs(EP_deconv)
		#HACK
		#low pass EP
		EP_deconv0=gaussian_filter1d(EP_deconv[0], sigma=sig_d)
		EP_deconv=EP_deconv0*(1-maskingPatterns)
	return EP_deconv, u1
