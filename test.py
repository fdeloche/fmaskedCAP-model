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


def plotMaskingAmountExcitations(BW10Func, maskingConditions, maskingIO, eps=1e-6, filter_model='gaussian', fmin=800, fmax=8000, axlist=None):
	'''
	Args:
		axlist:list of axes where to plot. If none creates a list of axes
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

	axlist2=[]
	for i, sq_exc in zip(range(maskingConditions.n_conditions), sq_excitations):
		ax= pl.subplot(maskingConditions.n_conditions//2, 2, i+1) if axlist is None else axlist[i]
		ax.set_title(maskingConditions.names[i], fontsize=10)
		#ax.plot(f, maskerSpectrum, '--')
		I=10*torch.log10(sq_exc+eps)
		ax.plot(f, maskingIO(I)*100)
		ax.set_xlabel('f')
		ax.set_ylabel('masking amount (%)')
		axlist2.append(ax)
	return axlist2