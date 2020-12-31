import torch
import numpy as np 
import matplotlib.pyplot as pl 

from masking import *
from latencies import *
from excitation import *
from deconv import *
from ur import *
from tuning import *


def plotMaskingExcitations(BW10Func, maskingConditions, filter_model='gaussian', fmin=800, fmax=8000, axlist=None):
	'''
	Args:
		axlist:list of axes where to plot. If none creates a list of axes
	Returns:
		the list of axes corresponding to the figures plotted
	'''
	m=500
	f=torch.linspace(fmin, fmax, m)
	sq_excitations = get_sq_masking_excitation_patterns_maskCond(f, BW10Func, maskingConditions, filter_model='gaussian')

	pl.suptitle('Masker spectra and excitations')

	maskerSpectra=torch.zeros(maskingConditions.n_conditions, m)
	amp_list, f_low_list, f_high_list = maskingConditions.get_tensor_lists()
	for amp, f_low, f_high in zip(amp_list, f_low_list, f_high_list):
		maskerSpectra+= torch.unsqueeze(amp, 1)*(torch.unsqueeze(f_low, 1)<f)*(torch.unsqueeze(f_high, 1)>f)

	axlist2=[]
	for i, maskerSpectrum, sq_exc in zip(range(maskingConditions.n_conditions), maskerSpectra, sq_excitations):
		ax= pl.subplot(maskingConditions.n_conditions//2, 2, i+1) if axlist is None else axlist[i]
		ax.set_title(maskingConditions.names[i], fontsize=10)
		ax.plot(f, maskerSpectrum, '--')
		ax.plot(f, torch.sqrt(sq_exc))
		ax.set_xlabel('f')
		axlist2.append(ax)
	return axlist2


def plotMaskingAmountExcitations(BW10Func, maskingConditions, maskingIO, eps=1e-6, filter_model='gaussian', fmin=800, fmax=8000, axlist=None, max_plots=8):
	'''
	Args:
		axlist:list of axes for the plots. If none creates a list of axes
	Returns:
		the list of axes corresponding to the figures plotted
	'''
	m=500
	f=torch.linspace(fmin, fmax, m)
	sq_excitations = get_sq_masking_excitation_patterns_maskCond(f, BW10Func, maskingConditions, filter_model='gaussian')

	pl.suptitle('Amount of masking')

	'''
	maskerSpectra=torch.zeros(maskingConditions.n_conditions, m)
	amp_list, f_low_list, f_high_list = maskingConditions.get_tensor_lists()
	for amp, f_low, f_high in zip(amp_list, f_low_list, f_high_list):
		maskerSpectra+= torch.unsqueeze(amp, 1)*(torch.unsqueeze(f_low, 1)<f)*(torch.unsqueeze(f_high, 1)>f)
	'''
	nb_plots=min(maskingConditions.n_conditions, max_plots)
	axlist2=[]
	for i, sq_exc in zip(range(maskingConditions.n_conditions), sq_excitations):
		if i==nb_plots:
			break
		ax= pl.subplot(nb_plots//2, 2, i+1) if axlist is None else axlist[i]
		ax.set_title(maskingConditions.names[i], fontsize=10)
		#ax.plot(f, maskerSpectrum, '--')
		I=10*torch.log10(sq_exc+eps)
		ax.plot(f, maskingIO(I)*100)
		ax.set_xlabel('f')
		ax.set_ylabel('masking amount (%)')
		ax.set_ylim([0, 100.])
		axlist2.append(ax)
	return axlist2



def plotExcitationPatterns(E, plot_raw_excitation=False, axlist=None, max_plots=6):
	'''
	Args:
		E:ExcitationPatterns object
		plot_raw_excitation: if True plot also raw excitation/amount of masking
		axlist:list of axes for the plots. If none creates a list of axes
	'''
	axlist2=[]
	if E.masked:
		maskAmounts, excs = E.get_tensors() 
		maskingConditions = E.maskingConditions
		if plot_raw_excitation:
			pl.suptitle('E_0, M  /  E_0*(1-M)')
		else:
			pl.suptitle('Excitation patterns: E_0*(1-M)')

		nb_plots=min(maskingConditions.n_conditions, max_plots)
		for i, maskAmount, exc in zip(range(maskingConditions.n_conditions), maskAmounts, excs):
			if i==nb_plots:
				break
			if plot_raw_excitation:
				ax= pl.subplot(nb_plots, 2, 2*i+1) if axlist is None else axlist[2*i]
			
				ax.plot(E.t*1e3, E.E0_nonmaskable, label='non maskable part')
				ax.plot(E.t*1e3, E.E0_maskable, label='maskable part')
				ax.legend()
				ax.twinx()
				ax.plot(E.t*1e3, maskAmount*100, label='masking Amount')
				ax.set_ylabel('Masking amount (%)')
				ax.set_xlabel('Time (ms)')
				ax.set_ylim([0, 100.])
				axlist2.append(ax)
				ax= pl.subplot(nb_plots, 2, 2*i+2) if axlist is None else axlist[2*i+1]
				
			else:	
				ax= pl.subplot(nb_plots, 2, i+1) if axlist is None else axlist[i]
	
			ax.set_title(maskingConditions.names[i], fontsize=10)
			ax.plot(E.t*1e3, exc)
			ax.set_xlabel('Time (ms)')

			if axlist is None:

				locs =torch.linspace(np.ceil(E.t[0]*1e3), np.floor(E.t[-1]*1e3), 10)
				ax2 = ax.twiny()
				ax2.set_xticks(locs)
				ax2.set_xticklabels([f'{CF/1e3:.1f}' for CF in list(E.latencies.f_from_t(locs*1e-3))])
				ax2.set_xlabel('Place: CF (kHz)')

			axlist2.append(ax)
		pl.tight_layout()
	else:
		ax = pl.gca() if axlist is None else axlist[0]
		ax.plot(E.t*1e3, E.E0_nonmaskable, label='non maskable part')
		ax.plot(E.t*1e3, E.E0_maskable, label='maskable part')


		ax.set_xlabel('Time (ms)')
		ax.legend()
		'''#no masking model --> no latencies
		if axlist is None:

			locs =torch.linspace(np.ceil(E.t[0]*1e3), np.floor(E.t[-1]*1e3), 10)
			ax2 = ax.twiny()
			ax2.set_xticks(locs)
			ax2.set_xticklabels([f'{CF/1e3:.1f}' for CF in list(E.latencies.f_from_t(locs*1e-3))])
			ax2.set_xlabel('Place: CF (kHz)')
		'''
		axlist2.append(ax)
	return axlist2



def plotSimulatedCAPs(E, u, axlist=None, shift=0):
	'''
	Args:
		E:ExcitationPatterns object
		u: unitary response (numpy array)
		axlist:list of axes for the plots. If none creates a list of axes
		shift:time shift for the convolution
	'''
	axlist2=[]
	if E.masked:
		excs = E.get_tensor() 
		maskingConditions = E.maskingConditions
		pl.suptitle('Simulated CAPs (+ excitation patterns)')
		for i, exc in zip(range(maskingConditions.n_conditions), excs):

			ax= pl.subplot(maskingConditions.n_conditions//2, 2, i+1) if axlist is None else axlist[2*i]
			ax.set_title(maskingConditions.names[i], fontsize=10)
			p=ax.plot(E.t*1e3, exc, linestyle='--', linewidth=1.5)
			ax2=ax.twinx()  if axlist is None else axlist[2*i+1]
			exc_np = exc.detach().numpy()
			CAP=np.convolve(exc_np, u, mode='full')
			t=E.t.numpy()
			ind_time=np.sum(t<(t[0]+shift))
			ind_time=min(ind_time, len(CAP)-len(E.t))
			CAP=CAP[ind_time:ind_time+len(E.t)]
			ax2.plot(E.t*1e3, CAP, color=p[0].get_color()) 
			ax2.grid(False)
			ax.set_xlabel('Time (ms)')
			axlist2.append(ax)
			axlist2.append(ax2)		
		pl.tight_layout()
	else:
		ax = pl.gca() if axlist is None else axlist[0]
		ax.plot(E.t*1e3, E.E0_nonmaskable, label='non maskable part', linestyle='--')
		p=ax.plot(E.t*1e3, E.E0_maskable, label='maskable part', linestyle='--', linewidth=1.5)
		E0=E.E0_nonmaskable+E.E0_maskable
		ax2=ax.twinx()  if axlist is None else axlist[1]
		exc_np = E0.detach().numpy()			
		CAP=np.convolve(exc_np, u, mode='full')
		t=E.t.numpy()
		ind_time=np.sum(t<(t[0]+shift))
		ind_time=min(ind_time, len(CAP)-len(E.t))
		CAP=CAP[ind_time:ind_time+len(E.t)]
		ax2.plot(E.t*1e3, CAP, color=p[0].get_color())
		ax2.grid(False)
		ax.set_xlabel('Time (ms)')
		ax.legend()
		axlist2.append(ax)
		axlist2.append(ax2)
	return axlist2
