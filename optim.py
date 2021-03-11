import numpy as np 
import torch 
import matplotlib.pyplot as pl 

from excitation import *
from latencies import *

def complex_multiplication(t1, t2):
	'''complex multiplication of a 2 D array by a (unsqueezed) 1D array (playing the role of the unitary response) '''
	real1, imag1 = t1[:, :, 0], t1[:, :, 1]
	real2, imag2 = t2[:, 0], t2[:, 1]
	return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim = -1)




def optim_steps(E, ur, signals_proc,  alpha_dic, nb_steps, n_dim_E0=7, plot_graphs=False):
	'''
	optimization steps on model parameters (parameters of masking I/O curves, raw excitation pattern, tuning) based on square error after convolution (based on RFFT/IRFFT). 
	NB: the excitations and convolution are computed at every step. ur is fixed.
	Args:
		E: excitation pattern object
		alpha_dic: dictionnary tensor (to be updated by gradient descent, requires_grad must be True) -> gradient descent step 
		signals_proc: real CAP signals, must be of the size of excitations
		nb_steps: number of gradient descent steps
		n_dim_E0: if E0_maskable is in alpha_dic, the gradient will be projected on the n_dim_E0 first dimensions of the Fourier basis (rfft)
		plot_graphs: monitors updated parameters	
	'''
	excs = E.get_tensor() 

	excs_fft = torch.rfft(excs, 1)
	ur_fft= torch.rfft(torch.tensor(ur), 1)
	CAPs_fft=complex_multiplication(excs_fft, ur_fft)
	CAPs = torch.irfft(CAPs_fft, 1, signal_sizes=(excs.shape[1], ))

	if E.E0_maskable in alpha_dic:
		#projection of gradient on first dimensions (Fourier basis)
		n_dim=n_dim_E0
		filter_fft=torch.zeros_like(torch.rfft(E.E0_maskable, 1))
		filter_fft[0:n_dim]=1

		def proj_fft2(grad):
			grad_fft=torch.rfft(grad, 1)
			grad_fft*=filter_fft
			return torch.irfft(grad_fft, 1, signal_sizes=grad.shape)


	for i in range(1, nb_steps+1):
		
		excs = E.get_tensor() 

		excs_fft = torch.rfft(excs, 1)
		#ur_fft= torch.rfft(torch.tensor(ur), 1)
		CAPs_fft=complex_multiplication(excs_fft, ur_fft)
		CAPs = torch.irfft(CAPs_fft, 1, signal_sizes=(excs.shape[1], ))

		err=torch.sum( (CAPs- torch.tensor(signals_proc) )**2 )
		
		err.backward()
		
		if E.E0_maskable in alpha_dic:
			alpha=alpha_dic[E.E0_maskable]
			E.E0_maskable.data -= alpha*proj_fft2(E.E0_maskable.grad)
			E.E0_maskable.grad.zero_()

			if plot_graphs:
				if isinstance(E.latencies, SingleLatency):
					lat=E.latencies
					pl.plot(np.linspace(lat.f_min*1e-3, lat.f_max*1e-3, 
						len(E.E0_maskable)), E.E0_maskable.detach().numpy(), label=f'step {i}')
					if i==1:
						pl.xlabel('Frequency (kHz)')
						pl.ylabel('raw excitation')
					
					if i==nb_steps+1:
						pl.legend()

		#TODO manage multiple plots


