import torch
import numpy as np
import matplotlib.pyplot as pl

import json 


class SigmoidMaskingDegreeFunction:
	'''Implements the sigmoid function
	f(I) = 1/(1+exp(-a(I-mu) ))
	'''
	def __init__(self, mu, a, requires_grad=False):
		'''
		Args:
			mu: point where 50% of max masking is reached
			a: shape parameter, 2x slope at I0 (considering that max masking is 100%)
		'''
		self.mu=torch.tensor(mu, requires_grad=requires_grad)
		self.a=torch.tensor(a, requires_grad=requires_grad)

	def __call__(self, I):
		return torch.sigmoid(self.a*(I-self.mu))

def get_masking_amount(mdFunc, sq_exc, eps=1e-6):
	'''
	Args:
		mdFunc: masking degree function
		sq_exc: squared excitation associated with masking patterns
	'''
	return mdFunc(10*torch.log10(eps+sq_exc))


class MaskingConditions:
	'''
	Masking conditions (representing Gaussian noises defined by bands)
	Attributes:
		n_bands: max number of bands
		n_conditions: number of masking conditions
		amp_list: torch tensors (dim: nb_conditions) of amplitudes by bands
		f_low_list: torch tensors (dim: nb_conditions) of low cut-off frequencies by bands
		f_high_list: torch tensors (dim: nb_conditions) of high cut-off frequencies by bands
	'''

	def __init__(stim_dic_list=[]):
		'''
		Args:
			stim_dic_list: list of (nested) dictionaries with items: n_bands, bands (amp, fc_low, fc_high)
		'''
		self.n_bands=0
		self.n_conditions=0
		#NB: private attributes below are not a list of tensors, but a list of list
		self._amp_list=[]
		self._f_low_list=[]
		self._f_high_list=[]
		self.add_conditions(stim_dic_list)

	def add_conditions(stim_dic_list):
		for stim_dic in stim_dic_list:

			stim_n_bands=stim_dic['n_bands']
			if stim_n_bands>self.n_bands:
				for k in range(n_bands, stim_n_bands):  #pad lists
					self._amp_list.append([0 for cond in range(self.n_conditions)])
					self._f_low_list.append([12500 for cond in range(self.n_conditions])
					self._f_high_list.append([13000 for cond in range(self.n_conditions)])
				self.n_bands=stim_n_bands

			for k in range(n_bands):
				band=stim_dic['bands'][k]
				try:
					amp=band['amp']
				except KeyError as e:
					amp=band['amplitude']
				self._amp_list[k].append(amp)
				self._f_low_list[k].append(band['fc_low'])
				self._f_high_list[k].append(band['fc_right'])				

			for k in range(stim_n_bands, self.n_bands):#pad lists
				self._amp_list[k].append(0)
				self._f_low_list[k].append(12500)
				self._f_high_list[k].append(13000)


			self.n_conditions+=1



	def add(stim_dic_list):
		add_conditions(stim_dic_list)


	@classmethod
	def from_json_files(cls, list_filenames):
		stim_dic_list=[]
		for filename in list_filenames:
		    with open(filename, 'r') as f:
		        stim_dict_list.append(json.loads(f.read()))

        return cls(stim_dic_list)

	@classmethod
	def from_json_strings(cls, list_strings):
		stim_dic_list=[]
		for st in list_strings:
	        stim_dict_list.append(json.loads(st))

        return cls(stim_dic_list)








def plotMaskingDegreeFunc(maskingDegreeFunction):
	I=torch.linspace(0, 100, 500)

	pl.figure()
	pl.plot(I, maskingDegreeFunction(I)*100)
	pl.xlabel('Spectral intensity (dB)')
	pl.ylabel('masking degree (%)')
	pl.ylim([0, 105])
	pl.show()
