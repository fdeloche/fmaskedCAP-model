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
		amp0: amplitude of reference (amp=1 defined in maskers means that amplitude is amp0) (default: 1). Taken in account when returning tensors
	'''

	def __init__(self, stim_dic_list=[]):
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
		self.names=[]
		self.amp0=1
		self.add_conditions(stim_dic_list)

	def add_conditions(self, stim_dic_list):
		self._modified=True #flag that forces the recomputation of tensors when get_tensor_lists is called
		for stim_dic in stim_dic_list:
			stim_n_bands=stim_dic['n_bands']
			if stim_n_bands>self.n_bands:
				for k in range(self.n_bands, stim_n_bands):  #pad lists
					self._amp_list.append([0 for cond in range(self.n_conditions)])
					self._f_low_list.append([12500 for cond in range(self.n_conditions)])
					self._f_high_list.append([13000 for cond in range(self.n_conditions)])
				self.n_bands=stim_n_bands

			for k in range(stim_n_bands):
				if stim_n_bands==1 and k==0 and not(isinstance(stim_dic['bands'], list)): #allows band given as 1 element as it is how is built by matlab
					band=stim_dic['bands']
				else:
					band=stim_dic['bands'][k]
				if 'amp' in band.items():
					amp=band['amp']
				else:
					amp=band['amplitude']
				self._amp_list[k].append(amp)
				self._f_low_list[k].append(band['fc_low'])
				self._f_high_list[k].append(band['fc_high'])				

			for k in range(stim_n_bands, self.n_bands):#pad lists
				self._amp_list[k].append(0)
				self._f_low_list[k].append(12500)
				self._f_high_list[k].append(13000)

			name=''
			if 'name' in stim_dic:
				name=stim_dic['name']
			self.names.append(name)

			self.n_conditions+=1


	def add(self, stim_dic_list):
		add_conditions(stim_dic_list)


	def add_json_strings(self,list_strings):
		stim_dic_list=[]
		for st in list_strings:
			stim_dic_list.append(json.loads(st))
		self.add_conditions(stim_dic_list)

	def set_amp0(self, amp0):
		self._modified=True
		self.amp0=amp0


	def set_amp0_dB(self, amp0dB):
		self._modified=True
		self.amp0=10**(amp0dB/20)


	def get_tensor_lists(self):
		'''
		Returns:
			tuple (amp_list, f_low_list, f_high_list)
		'''
		if self._modified:
			self._modified=False
			self._amp_tensor_list= [self.amp0*torch.tensor(l) for l in self._amp_list]
			self._f_low_tensor_list= [torch.tensor(l) for l in self._f_low_list]
			self._f_high_tensor_list= [torch.tensor(l) for l in self._f_high_list]
		return (self._amp_tensor_list, self._f_low_tensor_list, self._f_high_tensor_list)

	@property
	def amp_list(self):
		(res, _, _)=self.get_tensor_lists()
		return res

	@property
	def f_low_list(self):
		(_, res, _)=self.get_tensor_lists()
		return res
	
	@property
	def f_high_list(self):
		(_, _, res)=self.get_tensor_lists()
		return res
	
	@classmethod
	def from_json_files(cls, list_filenames):
		stim_dic_list=[]
		for filename in list_filenames:
			with open(filename, 'r') as f:
				stim_dic_list.append(json.loads(f.read()))

		return cls(stim_dic_list)

	@classmethod
	def from_json_strings(cls, list_strings):
		stim_dic_list=[]
		for st in list_strings:
			stim_dic_list.append(json.loads(st))

		return cls(stim_dic_list)

	def __repr__(self):
		st=f'Masking conditions\n nbs conditions: {self.n_conditions}\n n_bands: {self.n_bands}'
		if self.n_conditions<10:
			for cond in range(self.n_conditions):
				st+=f'\ncond {cond} {self.names[cond]}\n  ['
				for k in range(self.n_bands):
					amp=self._amp_list[k][cond]
					f_low=self._f_low_list[k][cond]
					f_high=self._f_high_list[k][cond]
					if k!=0:
						st+='; '
					st+=f'({amp}, {f_low} Hz, {f_high} Hz)'
				st+=f']'
		return st







def plotMaskingDegreeFunc(maskingDegreeFunction):
	I=torch.linspace(0, 100, 500)

	pl.figure()
	pl.plot(I, maskingDegreeFunction(I)*100)
	pl.xlabel('Spectral intensity (dB)')
	pl.ylabel('masking degree (%)')
	pl.ylim([0, 105])
	pl.show()
