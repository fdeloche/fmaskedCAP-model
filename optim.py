import numpy as np 
import torch 
import torch.fft
import torch.distributed as dist 
import matplotlib.pyplot as pl 
import matplotlib.colors as colors 

from excitation import *
from latencies import *

import rbf

'''
#not needed anymore
def complex_multiplication(t1, t2):
	#complex multiplication of a 2 D array by a (unsqueezed) 1D array (playing the role of the unitary response)
	real1, imag1 = t1[:, :, 0], t1[:, :, 1]
	real2, imag2 = t2[:, 0], t2[:, 1]
	return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim = -1)

'''


def optim_steps(E, ur, signals_proc,  alpha_dic, nb_steps, n_dim_E0=7, k_mode_E0=1, E0_t_min=0, E0_t_max=np.infty,
	sum_grad_E0=False,
	plot_E0_graph=False, plot_masking_I0_graph=False, 
	plot_Q10=False, fc_ref_Q10=0,
	step_plots=1, axes=None, ind_plots=None, step0=0, tot_steps=0, verbose=False, 
	Q10_distributed=False, E0_distributed=False, I0_distributed=False
	):
	'''
	optimization steps on model parameters (parameters of masking I/O curves, raw excitation pattern, tuning) based on square error after convolution (based on RFFT/IRFFT). 
	NB: the excitations and convolution are computed at every step. ur is fixed.
	Args:
		E: excitation pattern object
		alpha_dic: dictionnary of tensors (to be updated by gradient descent, requires_grad must be True) mapped to gradient descent step 
		signals_proc: real CAP signals, must be of the size of excitations
		nb_steps: number of gradient descent steps
		sum_grad_E0: if True and E.E0_maskable in alpha_dic, apply gradient descent only on the mean amplitude (by summing gradients) [note: normally not needed as now the model includes a paramter E0_maskable_amp]
		n_dim_E0: if E0_maskable is in alpha_dic, the gradient will be projected on the n_dim_E0 first dimensions of the Fourier basis (rfft)
		k_mode_E0: if E0_maskable is in alpha_dic, allows for the projection to be on the harmonics of the k-th mode
		E0_t_min: (not applicable to single latency model or when use_bincount is True), if E0_maskable is in alpha_dic, restricts E0_maskable to [t_min, t_max]
		E0_t_max: (not applicable to single latency model or when use_bincount is True),), if E0_maskable is in alpha_dic, restricts E0_maskable to [t_min, t_max]
		plot_graphs: monitors updated parameters
		step_plots: plot every step_plots steps 
		axes: list of axes on which the figures are plotted. If None (default), creates the axes
		ind_plots: dictionnary for the subplots
		step0: ref for step 0 (default: 0)
		tot_steps: total number of steps (if optim_steps is called several times), for plotting purposes onely
		Q10_distributed: if True, forwards gradients to main node for Q10 (then encoded by a RBF net)  [note: asynchronous communications]
		E0_distributed: if True, forwards gradients to main node for E0 (note: still applies projection to gradient if applicable)
		I0_distributed: if True, forwards gradients to main node for I0 (for weibull cdf, encoded by a RBF net)
		verbose: prints error at end of optim steps
		fc_ref_Q10: CF (ref frequency for Q10). also used for plotting i/o func if dependent on frequency
	Returns:
		axes: list of pyplot axes if plots
		ind_plots: dictionnary for the subplots
		err_list: list of errors across iterations
	'''
	assert (axes is None) == (ind_plots is None)

	if plot_Q10:
		assert fc_ref_Q10>0, "fc_ref_Q10 must be set when plot_Q10 is True"

	if tot_steps==0:
		tot_steps=nb_steps
		
	cdict = {'red':   ((0.0,  0.22, 0.0),
			   (0.5,  1.0, 1.0),
			   (1.0,  0.89, 1.0)),

	 'green': ((0.0,  0.49, 0.0),
			   (0.5,  1.0, 1.0),
			   (1.0,  0.12, 1.0)),

	 'blue':  ((0.0,  0.72, 0.0),
			   (0.5,  0.0, 0.0),
			   (1.0,  0.11, 1.0))}

	cmap = colors.LinearSegmentedColormap('custom', cdict)

	err_list=[]


	if E.E0_maskable in alpha_dic:
		#projection of gradient on first dimensions (Fourier basis)
		n_dim=n_dim_E0
		filter_fft=torch.zeros_like(torch.fft.rfft(E.E0_maskable))
		filter_fft[0:(n_dim*k_mode_E0):k_mode_E0]=1

		def proj_fft2(grad):
			grad_fft=torch.fft.rfft(grad)
			grad_fft*=filter_fft
			return torch.fft.irfft(grad_fft, n=len(grad) )

		if not(isinstance(E.latencies, SingleLatency) or E.use_bincount):
			filter_t=(E.t<E0_t_max)*(E.t>E0_t_min)
			def proj_E0(E0):
				return E0*filter_t

	if ind_plots is None:
		nb_plots=sum([plot_E0_graph, plot_masking_I0_graph, plot_Q10])
	else:
		nb_plots=len(ind_plots)

	if ind_plots is None:
		ind_plots2={}
		axes2=[]

	for i in range(1, nb_steps+1):
		step=step0+i
		cstep=cmap(step/tot_steps)
		excs = E.get_tensor() 

		excs_fft = torch.fft.rfft(excs)
		if i==1:
			ur_fft= torch.fft.rfft(torch.tensor(ur))
		#CAPs_fft=complex_multiplication(excs_fft, ur_fft)
		CAPs_fft=excs_fft*ur_fft
		
		CAPs = torch.fft.irfft(CAPs_fft, n=excs.shape[1])

		err=torch.sum( (CAPs- torch.tensor(signals_proc) )**2 )
		
		err_list.append(err)

		if i==nb_steps and verbose:
			print(f"step {step}, err RMS: {torch.sqrt(err/(E.maskingConditions.n_conditions*len(E.t)))*1e3:.4f} (µV)")
		err.backward()
		
		for tensor in alpha_dic:
			if tensor.data_ptr() == E.E0_maskable.data_ptr():

				alpha=alpha_dic[tensor]
				if E0_distributed:
					grad=proj_fft2(E.E0_maskable.grad)
					try:
						hand = dist.isend( grad, 0, tag=2000+i)
						hand.wait()
					except RuntimeError as e:
						print(e)
						print(f'handle send grad E0 it {i} (step {step}) not completed before timeout')

				if sum_grad_E0:
					E.E0_maskable.data = (1-alpha*torch.sum(E.E0_maskable.grad))*E.E0_maskable.data
				else:
					E.E0_maskable.data -= alpha*proj_fft2(E.E0_maskable.grad)
				#E.E0_maskable.grad.zero_()
				if not(isinstance(E.latencies, SingleLatency) or E.use_bincount):
					E.E0_maskable.data= proj_E0(E.E0_maskable)
			else:
				alpha=alpha_dic[tensor]
				tensor.data -= alpha*tensor.grad
				#tensor.grad.zero_()

			if Q10_distributed and tensor.data_ptr() == E.bw10Func.Q10RBFnet.l2.weight.data_ptr():
				#forwards gradient (in addition to updating 'local' params)
				try:
					hand = dist.isend( tensor.grad, 0, tag=1000+i)
					hand.wait()
				except RuntimeError as e:
					print(e)
					print(f'handle send grad Q10 it {i} (step {step}) not completed before timeout')


			if I0_distributed and tensor.data_ptr() == E.maskingIOFunc.rbfNet.l2.weight.data_ptr():
				#forwards gradient (in addition to updating 'local' params)
				try:
					hand = dist.isend( tensor.grad, 0, tag=3000+i)
					hand.wait()
				except RuntimeError as e:
					print(e)
					print(f'handle send grad RBFbet I0 it {i} (step {step}) not completed before timeout')




		for tensor in E.list_param_tensors():
			if tensor.requires_grad and tensor.grad is not None:
				tensor.grad.zero_()

		ind_plot=0

		if plot_E0_graph:
			if ind_plots is None:
				ind_plot+=1
				if i==1:
					ind_plots2['E0']=ind_plot
			else:
				ind_plot=ind_plots['E0']
			if i==1:
				if axes is None:
					ax1 = pl.subplot(nb_plots, 1, ind_plot)
					axes2.append(ax1)
				else:
					ax1 = axes[ind_plot-1]
			if isinstance(E.latencies, SingleLatency) or E.use_bincount:
				if isinstance(E.latencies, SingleLatency):
					lat=E.latencies
					f=np.linspace(lat.f_min*1e-3, lat.f_max*1e-3, 
						len(E.E0_maskable))
				else:
					f=E.bincount_f

				if (i-1)%step_plots==0:
					ax1.plot(f , E.E0_maskable.detach().numpy(), label=f'step {step}', color=cstep)
					
				if i==1 and axes is None:

					ax1.set_title('Raw excitation pattern')
					ax1.set_xlabel('Frequency (kHz)')
					ax1.set_ylabel('raw excitation')
				
				if i==nb_steps:
					#ax1.legend()

					ax1.set_ylim([-0.2, 1.4])
					pass
			else:
				if (i-1)%step_plots==0:
					ax1.plot( E.t*1e3 , E.E0_maskable.detach().numpy(), label=f'step {step}', color=cstep)
					
				if i==1 and axes is None:

					ax1.set_title('Raw excitation pattern')
					ax1.set_xlabel('Time (ms)')
					ax1.set_ylabel('raw excitation')
				
				if i==nb_steps:
					#ax1.legend()
					pass


		if plot_masking_I0_graph:

			if ind_plots is None:
				ind_plot+=1
				if i==1:
					ind_plots2['I0']=ind_plot
			else:
				ind_plot=ind_plots['I0']
			if i==1:
				I=np.linspace(-30, 20)
				if axes is None:
					ax2 = pl.subplot(nb_plots, 1, ind_plot)
					axes2.append(ax2)
				else:
					ax2 = axes[ind_plot-1]

			if (i-1)%step_plots==0:
				ax2.plot(I, E.maskingIOFunc(torch.tensor(I), 
					torch.tensor(fc_ref_Q10, dtype=torch.float32)).detach().numpy(), label=f'step {step}', color=cstep)
			
			if i==1 and axes is None:
				ax2.set_title('Masking IO Function')
				ax2.set_xlabel('Power spectral density (dB)')
				ax2.set_ylabel('masking (max: broadband)')

			if i==nb_steps:
				#ax2.legend()
				pass

		if plot_Q10:
			if ind_plots is None:
				ind_plot+=1
				if i==1:
					ind_plots2['Q10']=ind_plot
			else:
				ind_plot=ind_plots['Q10']
			if i==1:
				if axes is None:
					ax3 = pl.subplot(nb_plots, 1, ind_plot)
					axes2.append(ax3)
				else:
					ax3 = axes[ind_plot-1]

			with torch.no_grad():
				ax3.plot(step, fc_ref_Q10/E.bw10Func(torch.tensor(fc_ref_Q10, dtype=torch.float32)), '+', color=cstep)
			

			if i==1 and axes is None:
				ax3.set_title('Filter tuning')
				ax3.set_xlabel('Step')
				ax3.set_ylabel('Q10')
			if i==1:
				ax3.set_xlim(right=step+nb_steps)
	if ind_plots is None:
		ind_plots=ind_plots2
		axes=axes2
	return axes, ind_plots, err_list



