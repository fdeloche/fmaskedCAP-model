import torch
import numpy as np 
import matplotlib.pyplot as pl 

from masking import *
from latencies import *
from excitation import *
from deconv import *
from ur import *
from tuning import *
import re



def plotMaskingExcitations(BW10Func, maskingConditions, filter_model='gaussian', fmin=800, fmax=8000, 
	axlist=None, reg_ex=None, freq_factor=1):
	'''
	Args:
		axlist:list of axes where to plot. If none creates a list of axes
		freq_factor: computes excitation with f*freq_factor (default: 1)
	Returns:
		the list of axes corresponding to the figures plotted
	'''
	m=500
	f=torch.linspace(fmin, fmax, m)
	sq_excitations = get_sq_masking_excitation_patterns_maskCond(freq_factor*f, BW10Func, maskingConditions, filter_model=filter_model)

	pl.suptitle('Masker spectra and excitations')

	maskerSpectra=torch.zeros(maskingConditions.n_conditions, m)
	amp_list, f_low_list, f_high_list = maskingConditions.get_tensor_lists()
	for amp, f_low, f_high in zip(amp_list, f_low_list, f_high_list):
		maskerSpectra+= torch.unsqueeze(amp, 1)*(torch.unsqueeze(f_low, 1)<f)*(torch.unsqueeze(f_high, 1)>f)

	axlist2=[]
	ind=0
	for i, maskerSpectrum, sq_exc in zip(range(maskingConditions.n_conditions), maskerSpectra, sq_excitations):
		if not reg_ex is None:
			if not(re.match(reg_ex, maskingConditions.names[i])):
				continue 
		ax= pl.subplot(maskingConditions.n_conditions//2, 2, ind+1) if axlist is None else axlist[i]
		ax.set_title(maskingConditions.names[i], fontsize=10)
		ax.plot(f, maskerSpectrum, '--')
		ax.plot(f, torch.sqrt(sq_exc))
		ax.set_xlabel('f')
		axlist2.append(ax)
		ind+=1
	return axlist2


def plotMaskingAmountExcitations(BW10Func, maskingConditions, maskingIO, eps=1e-6, filter_model='gaussian', fmin=800, fmax=8000, 
	suppressionAmount=None, refMaskers=None, axlist=None, max_plots=8, reg_ex=None):
	'''
	Args:
		refMaskers: masking Conditions (same number of conds as maskingConditions) serving as reference maskers. 
		axlist:list of axes for the plots. If none creates a list of axes
	Returns:
		the list of axes corresponding to the figures plotted
	'''
	m=500
	f=torch.linspace(fmin, fmax, m)
	sq_excitations = get_sq_masking_excitation_patterns_maskCond(f, BW10Func, maskingConditions, filter_model=filter_model)
	if not(refMaskers is None):
		sq_excitations_ref = get_sq_masking_excitation_patterns_maskCond(f, BW10Func, refMaskers, filter_model=filter_model)


	if suppressionAmount is not None:
		suppAmount=suppressionAmount(f, maskingConditions)
		if not(refMaskers is None):
			suppAmountRefMaskers=suppressionAmount(f, refMaskers)

	pl.suptitle('Amount of masking')

	'''
	maskerSpectra=torch.zeros(maskingConditions.n_conditions, m)
	amp_list, f_low_list, f_high_list = maskingConditions.get_tensor_lists()
	for amp, f_low, f_high in zip(amp_list, f_low_list, f_high_list):
		maskerSpectra+= torch.unsqueeze(amp, 1)*(torch.unsqueeze(f_low, 1)<f)*(torch.unsqueeze(f_high, 1)>f)
	'''
	nb_plots=min(maskingConditions.n_conditions, max_plots)
	axlist2=[]
	ind=0
	for i, sq_exc in zip(range(maskingConditions.n_conditions), sq_excitations):
		if ind==nb_plots:
			break
		if not reg_ex is None:
			if not(re.match(reg_ex, maskingConditions.names[i])):
				continue 
		ax= pl.subplot(nb_plots//2, 2, ind+1) if axlist is None else axlist[i]
		ax.set_title(maskingConditions.names[i], fontsize=10)
		#ax.plot(f, maskerSpectrum, '--')
		I=10*torch.log10(sq_exc+eps)
		I2 = I if suppressionAmount is None else I - suppAmount[i]
		if not(refMaskers is None):
			sq_exc_ref=sq_excitations_ref[i]
			Iref=10*torch.log10(sq_exc_ref+eps)
			I2ref =  Iref if suppressionAmount is None else Iref - suppAmountRefMaskers[i]

			#ax.plot(f, maskingIO(I2, f)*100, label='masking amount')

			#ax.plot(f, maskingIO(I2ref, f)*100, label='masking amount')
			ax.plot(f, (maskingIO(I2, f)-maskingIO(I2ref, f))*100, label='masking amount')
			ax.set_ylim([-100, 100.])
		else:
			ax.plot(f, maskingIO(I2, f)*100, label='masking amount')
			ax.set_ylim([0, 100.])
		ax.set_xlabel('f')
		ax.set_ylabel('Masking amount (%)')
		axlist2.append(ax)
		if not(suppressionAmount is None):
			ax.twinx()
			ax.plot(f, suppAmount[i], label='suppression amount', linestyle='--')
			if not(refMaskers is None):
				ax.plot(f, suppAmountRefMaskers[i], label='suppression amount', linestyle='--')
			#ax.set_ylabel('Suppression amount (dB)')
		ind+=1
	return axlist2



def plotExcitationPatterns(E, plot_raw_excitation=False, axlist=None, max_plots=6, reg_ex=None, ylim_top=None):
	'''
	Args:
		E:ExcitationPatterns object
		plot_raw_excitation: if True plot also raw excitation/amount of masking
		axlist:list of axes for the plots. If none creates a list of axes
	'''
	axlist2=[]
	if E.masked:
		if isinstance(E.latencies, SingleLatency) or E.use_bincount:
			return plotExcitationPatternsSingleLat(E, plot_raw_excitation=plot_raw_excitation, axlist=axlist, max_plots=max_plots, 
				reg_ex=reg_ex, ylim_top=ylim_top)
		maskAmounts, excs = E.get_tensors() 
		maskingConditions = E.maskingConditions
		if plot_raw_excitation:
			pl.suptitle('E_0, M  /  E_0*(1-M)')
		else:
			pl.suptitle('Excitation patterns: E_0*(1-M)')

		nb_plots=min(maskingConditions.n_conditions, max_plots)
		ind=0
		for i, maskAmount, exc in zip(range(maskingConditions.n_conditions), maskAmounts, excs):
			if ind==nb_plots:
				break
			if not reg_ex is None:
				if not(re.match(reg_ex, maskingConditions.names[i])):
					continue 
			if plot_raw_excitation:
				ax= pl.subplot(nb_plots, 2, 2*ind+1) if axlist is None else axlist[2*i]
			
				ax.plot(E.t*1e3, E.E0_nonmaskable, label='non maskable part')
				ax.plot(E.t*1e3, E.E0_maskable, label='maskable part')
				ax.legend()
				ax.twinx()
				ax.plot(E.t*1e3, maskAmount, label='masking Amount')
				ax.set_ylabel('Masking amount')
				ax.set_xlabel('Time (ms)')
				ax.set_ylim([0, 1.])
				axlist2.append(ax)
				ax= pl.subplot(nb_plots, 2, 2*ind+2) if axlist is None else axlist[2*i+1]
				
			else:	
				ax= pl.subplot(nb_plots, 2, ind+1) if axlist is None else axlist[i]
	
			ax.set_title(maskingConditions.names[i], fontsize=10)
			ax.plot(E.t*1e3, exc)
			ax.set_xlabel('Time (ms)')

			if axlist is None:

				locs =torch.arange(np.ceil((E.t[0]+1e-4)*1e3), np.floor((E.t[-1]-1e-4)*1e3)+1)
				ax2 = ax.twiny()
				ax2.plot(E.t*1e3, -np.ones_like(E.t)) #HACK
				ax2.set_xticks(locs)
				ax2.set_xticklabels([f'{CF/1e3:.1f}' for CF in list(E.latencies.f_from_t(locs*1e-3))])
				ax2.set_xlabel('Place: CF (kHz)')

				ax2.set_ylim(bottom=0)
				if ylim_top is not None:
					ax2.set_ylim(top=ylim_top)
				
			axlist2.append(ax)
			ind+=1
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





def plotExcitationPatternsSingleLat(E, plot_raw_excitation=False, axlist=None, max_plots=6, reg_ex=None, ylim_top=None):
	'''
	Aux function for excitations with a single latency (raw excitation defined in frequency) or computed with bincount
	Args:
		E:ExcitationPatterns object
		plot_raw_excitation: if True plot also raw excitation/amount of masking
		axlist:list of axes for the plots. If none creates a list of axes
	'''
	axlist2=[]
	assert (E.masked and (isinstance(E.latencies, SingleLatency) or E.use_bincount))
	maskAmounts, excs = E.get_tensors() 
	maskingConditions = E.maskingConditions
	if plot_raw_excitation:
		pl.suptitle('E_0, M  /  E_0*(1-M)')
	else:
		pl.suptitle('Excitation patterns: E_0*(1-M)')

	nb_plots=min(maskingConditions.n_conditions, max_plots)
	ind=0
	if isinstance(E.latencies, SingleLatency):
		f=E.latencies.get_f_linspace( len(E.E0_maskable))
	else:
		f=E.bincount_f
	for i, maskAmount, exc in zip(range(maskingConditions.n_conditions), maskAmounts, excs):
		if ind==nb_plots:
			break
		if not reg_ex is None:
			if not(re.match(reg_ex, maskingConditions.names[i])):
				continue 
		if plot_raw_excitation:
			ax= pl.subplot(nb_plots, 2, 2*ind+1) if axlist is None else axlist[2*i]
		
			ax.plot(f, E.E0_nonmaskable, label='non maskable part')
			ax.plot(f, E.E0_maskable, label='maskable part')
			ax.legend()
			ax.twinx()
			ax.plot(f, maskAmount, label='masking Amount')
			ax.set_ylabel('Masking amount')
			ax.set_xlabel('Frequency (Hz)')
			ax.set_ylim([0, 1.])
			axlist2.append(ax)
			ax= pl.subplot(nb_plots, 2, 2*ind+2) if axlist is None else axlist[2*i+1]
			
		else:	
			ax= pl.subplot(nb_plots, 2, ind+1) if axlist is None else axlist[i]

		ax.set_title(maskingConditions.names[i], fontsize=10)
		ax.plot(E.t*1e3, exc)
		ax.set_xlabel('Time (ms)')

		if axlist is None:

			ax.set_ylim(bottom=0)
			if ylim_top is not None:
				ax.set_ylim(top=ylim_top)
			
		axlist2.append(ax)
		ind+=1
	pl.tight_layout()

	return axlist2

def plotSimulatedCAPs(E, u=None, CAParray=None, axlist=None, shift=0, max_plots=8, ylim=None, reg_ex=None, title='Simulated CAPs (+ excitation patterns)', plot_excitations=True, plotargs={}):
	'''
	Args:
		E:ExcitationPatterns object
		u: unitary response (numpy array)
		CAParray: array of CAP signals (if the convolution is done outside the function), must be of size (nb_conditions, len(E.t)) . either CAParray or u must be given
		axlist:list of axes for the plots. If none creates a list of axes
		shift:time shift for the convolution
		ylim: interval to pass to matplotlib (opt.)
		reg_ex: regular expression to filter masker names (opt.)
	'''
	assert not(u is None) or not(CAParray is None), 'either CAParray or u must be given'
	axlist2=[]
	if E.masked:
		excs = E.get_tensor() 
		maskingConditions = E.maskingConditions
		pl.suptitle(title)
		nb_plots=min(maskingConditions.n_conditions, max_plots)
		ind=0
		for i, exc in zip(range(maskingConditions.n_conditions), excs):
			if ind==nb_plots:
				break
			if not reg_ex is None:
				if not(re.match(reg_ex, maskingConditions.names[i])):
					continue 
			ax= pl.subplot((nb_plots+1)//2, 2, ind+1) if axlist is None else axlist[2*i]
			ax.set_title(maskingConditions.names[i], fontsize=10)
			if plot_excitations:
				p=ax.plot(E.t*1e3, exc, linestyle='--', linewidth=1.5, **plotargs)
				if len(plotargs)==0:
					plotargs= {"color":p[0].get_color()}
			ax2=ax.twinx()  if axlist is None else axlist[2*i+1]

			if not CAParray is None:
				CAP=CAParray[i]
				ax2.plot(E.t*1e3, CAP, **plotargs) 
				ax2.grid(False)
			else:
				exc_np = exc.detach().numpy()
				CAP=np.convolve(exc_np, u, mode='full')
				t=E.t.numpy()
				ind_time=np.sum(t<(t[0]+shift))
				ind_time=min(ind_time, len(CAP)-len(E.t))
				CAP=CAP[ind_time:ind_time+len(E.t)]
				ax2.plot(E.t*1e3, CAP,  **plotargs) 
			ax2.grid(False)
			ax.set_xlabel('Time (ms)')
			axlist2.append(ax)
			axlist2.append(ax2)		
			if not ylim is None:
				pl.ylim(ylim)
			ind+=1
		pl.tight_layout()

	else:
		ax = pl.gca() if axlist is None else axlist[0]
		ax.plot(E.t*1e3, E.E0_nonmaskable, label='non maskable part', linestyle='--')
		if plot_excitations:
			p=ax.plot(E.t*1e3, E.E0_maskable, label='maskable part', linestyle='--', linewidth=1.5, **plotargs)
			if len(plotargs)==0:
				plotargs= {"color":p[0].get_color()}		
		E0=E.E0_nonmaskable+E.E0_maskable
		ax2=ax.twinx()  if axlist is None else axlist[1]
		exc_np = E0.detach().numpy()			
		CAP=np.convolve(exc_np, u, mode='full')
		t=E.t.numpy()
		ind_time=np.sum(t<(t[0]+shift))
		ind_time=min(ind_time, len(CAP)-len(E.t))
		CAP=CAP[ind_time:ind_time+len(E.t)]
		ax2.plot(E.t*1e3, CAP,  **plotargs)
		ax2.grid(False)
		ax.set_xlabel('Time (ms)')
		ax.legend()
		axlist2.append(ax)
		axlist2.append(ax2)
	return axlist2
