import torch
import numpy as np
import matplotlib.pyplot as pl

import json 
import rbf

from scipy.optimize import curve_fit


###########  Masking I/O functions ###########

class SigmoidIOFunc:
	'''Implements the sigmoid function
	f(I) = 1/(1+exp(-a(I-mu) ))
	'''
	def __init__(self, mu, a, max_masking=1., requires_grad=False, constrained_at_Iref=False, Iref=-20):
		'''
		Args:
			mu: point where 50% of max masking is reached
			a: shape parameter, 2x slope at I0 (considering that max masking is 100%)
			constrained_at_Iref: if True, constrains the function to equal 1 at Iref.  (in this case, mmax is superfluous)
			Iref: Iref in dB in the case of 'constrained_at_Iref
		'''

		self.mu=torch.tensor(mu, requires_grad=requires_grad)
		self.a=torch.tensor(a, requires_grad=requires_grad)

		self.constrained_at_Iref=constrained_at_Iref
		self._Iref=Iref
		
		self.mmax=torch.tensor(max_masking, requires_grad=requires_grad)


	def __call__(self, I, f=0):
		if self.constrained_at_Iref:
			return torch.sigmoid(self.a*(I-self.mu))/torch.sigmoid(self.a*(self._Iref-self.mu))
		else:
			return self.mmax*torch.sigmoid(self.a*(I-self.mu))

	def list_param_tensors(self):
		return [self.mu, self.a, self.mmax]

	def fit_data(self, I_values, m_values, init_with_new_values=True, set_mmax=False, constrained_at_Iref=False, Iref=-20, method='lm'):
		'''
		Sets mu and a to fit I_values (np array) and m_values (np array, masking amount values, max 100%).
		Levenberg-Maquardt algorithm or dog leg method (max mu:200, see in code).  [method: 'lm' or 'dogbox'] 
		Args:
			init_with_new_values: if True, initialization of algorithm with (median value I, 4/(max(I) - min(I)), if false init with self.a, self.mu
			set_mmax: if maximum masking is a free parameter (if False, set to 1)
			constrained_at_Iref: if True, constrains the function to equal 1 at Iref. (if True, set_mmax must be False)
			Iref: Iref in dB in the case of 'constrained_at_Iref'
			'''

		Iref=np.array([Iref])

		def aux_jac(x, mu, a):
			y=a*(x-mu)
			sig2=aux_f(x, mu, a)**2
			temp=-sig2*np.exp(-y)
			df_mu=temp*a
			df_a=temp*(-(x-mu))
			return np.stack((df_mu, df_a), axis=1)

		def aux_f(x, mu, a):
			y=a*(x-mu)
			return 1/(1+np.exp(-y))


		#if mmax is not set to 1
		def aux_jac2(x, mu, a, mmax):
			y=a*(x-mu)
			auxf=aux_f(x, mu, a)
			sig2=auxf**2
			temp=-sig2*np.exp(-y)
			df_mu=temp*a
			df_a=temp*(-(x-mu))
			df_mmax=auxf
			return np.stack((mmax*df_mu, mmax*df_a, df_mmax), axis=1)

		def aux_f2(x, mu, a, mmax):
			y=a*(x-mu)
			return mmax/(1+np.exp(-y))


		#if constrained_at_Iref

		def aux_jac3(x, mu, a):
			sigm_ref=aux_f(Iref, mu, a)
			return 1/sigm_ref*aux_jac(x, mu, a) - aux_f(x, mu, a)[:, None]/sigm_ref**2*aux_jac(Iref, mu, a)

		def aux_f3(x, mu, a):
			return aux_f(x, mu, a)/aux_f(Iref, mu, a)


		if init_with_new_values:
			p0 = (np.median(I_values), 4/(np.amax(I_values) - np.amin(I_values) ))
		else:
			p0= (self.mu.numpy(), self.a.numpy())

		if set_mmax:

			if init_with_new_values:
				p0=p0+(1,)
			else:
				p0=p0+(self.mmax.numpy(),)

		if constrained_at_Iref:
			if method=='dogbox':
				params, _= curve_fit(aux_f3, I_values, m_values,
					p0= p0, method=method, jac=aux_jac3, bounds=([-100, 0], [200, np.infty]))
			else:
				params, _= curve_fit(aux_f3, I_values, m_values,
					p0= p0, method=method, jac=aux_jac3)
			mmax=1/aux_f(Iref[0], params[0], params[1])
			self.mmax.data=torch.tensor(mmax)
			print(f'fitting data (constraint =1 at I={Iref[0]:.1f}dB) :\n mu={params[0]:.2f}, a={params[1]:.4f}, mmax:{mmax:.3f}')
		elif set_mmax:
			params, _= curve_fit(aux_f2, I_values, m_values,
				p0= p0, method='lm', jac=aux_jac2)
			print(f'fitting data:\n mu={params[0]:.2f}, a={params[1]:.4f}, mmax:{params[2]:.3f}')
		else:
			params, _= curve_fit(aux_f, I_values, m_values,
				p0= p0, method='lm', jac=aux_jac)
			print(f'fitting data:\n mu={params[0]:.2f}, a={params[1]:.4f}')
		self.mu.data=torch.tensor(params[0])
		self.a.data=torch.tensor(params[1])
		if set_mmax:
			self.mmax.data=torch.tensor(params[2])

	@classmethod
	def from_datapts(cls, I_values, m_values, requires_grad=False):
		'''
		Returns an object of the class and fits the parameter to (I_values, m_values) (see method fit_data)
		'''
		sig=cls(0,0, requires_grad=requires_grad)
		sig.fit_data(I_values, m_values)
		return sig

	def write_to_npz(self, filename):
		def get_data(t):
			return t.detach().numpy()

		np.savez(filename, mu=get_data(self.mu),
			a=get_data(self.a), Iref=self._Iref, mmax=get_data(self.mmax),
			constrained_at_Iref=self.constrained_at_Iref)

	@classmethod
	def load_from_npz(cls, filename, requires_grad=False):
		with np.load(filename) as f:
			mu=f['mu']
			a=f['a']
			mmax=f['mmax']
			Iref=f['Iref']
			constrained_at_Iref=f['constrained_at_Iref']
		return cls(mu, a, max_masking=mmax, requires_grad=requires_grad,
			constrained_at_Iref=constrained_at_Iref, Iref=Iref)








SigmoidMaskingDegreeFunction=SigmoidIOFunc  #old name



class WeibullCDF_IOFunc:
	'''Implements the Weibull CDF function
	f(I) = 1-exp(- ((I-I0)/scale)^k )
	'''
	def __init__(self, I0=0., scale=40., k=10., requires_grad=False, mmax=1., constrained_at_Iref=False, Iref=-20):
		'''
		Args:
			I0: localization parameter (max intensity associated with 0% masking)
			scale: scale parameter (63% masking intensity reached at I0+scale)
			k: shape parameter
			mmax: maximum masking
			constrained_at_Iref: if True, constrains the function to equal 1 at Iref.  (in this case, mmax is superfluous)
			Iref: Iref in dB in the case of 'constrained_at_Iref'
		'''
		self.I0=torch.tensor(I0, requires_grad=requires_grad)
		self.constant_I0=True #does not depend on f by default		
		self.scale=torch.tensor(scale, requires_grad=requires_grad)	
		self.k=torch.tensor(k, requires_grad=requires_grad)

		self.constrained_at_Iref=constrained_at_Iref
		self._Iref=Iref
		
		self.mmax=torch.tensor(mmax, requires_grad=requires_grad)


	def set_I0_w_RBFNet(self, rbfNet, plus_lambda=False):
		'''set I0 as a function of f with a rbf network
		Args:
			rbfNet: RBF network, see rbf.py
			plus_lambda: if True, the RBF network output is considered to be (I0+scale)  (63% of max value whatever the value of k)'''
		self.constant_I0=False
		self.rbfNet=rbfNet
		self.plus_lambda=plus_lambda



	def __call__(self, I, f=torch.tensor([0.]) ):
		if self.constant_I0:
			I0=self.I0
		else:
			if len(f.shape)==0:
				f=torch.unsqueeze(f, -1)
			I0=self.rbfNet(f)
			if self.plus_lambda:
				I0-=self.scale
			#I0=torch.squeeze(I0, -1)
			#I0=torch.unsqueeze(I0, 0)
			if len(I.shape)>1:
				I0=torch.transpose(I0, 0, 1) #out_dim (1) becomes batch dim
			else:
				I0=torch.squeeze(I0, -1)
		Delta_I=torch.maximum((I-I0), torch.tensor(0.))
		if self.constrained_at_Iref:
			Delta_I_ref=torch.maximum((self._Iref-I0), torch.tensor(0.))
			return (1-torch.exp( -(Delta_I/self.scale)**self.k))/(1-torch.exp( -(Delta_I_ref/self.scale)**self.k))
		else:
			return self.mmax*(1-torch.exp( -(Delta_I/self.scale)**self.k))


	def list_param_tensors(self):
		params=[self.scale, self.k, self.mmax]
		if self.constant_I0:
			params.append(self.I0)
		else:
			params.append(self.rbfNet.l2.weight)
		return params

	def fit_data(self, I_values, m_values, init_with_new_values=True,  constrained_at_Iref=False, Iref=-20):
		'''
		Sets I0, scale and k to fit I_values (np array) and m_values (np array, masking amount values, max 100%).
		Dog leg method (based on Levenberg-Maquardt algorithm, max k=20). 
		Note: fitting mmax is not implemented (but it is automatically set if constrained_at_Iref is True)
		Args:
			init_with_new_values: if True, initialization of algorithm with (min I, median I - I0, 2) , if false init with values defined by class init
			constrained_at_Iref: if True, constrains the function to equal 1 at Iref.
			Iref: Iref in dB in the case of 'constrained_at_Iref'
		'''

		def aux_f(I, I0, scale, k):

			Delta_I=np.maximum((I-I0), 0 )
			return 1-np.exp( -(Delta_I/scale)**k)

		def aux_jac(I, I0, scale, k):
			Delta_I=np.maximum((I-I0), 0 )
			xx=Delta_I/scale
			temp=np.exp( -xx**k)
			df_I0=-temp*k*xx**(k-1)*1/scale
			df_sc=-temp*xx**k*k*1/scale
			df_k=temp*xx**k*np.log(xx+1e-6)
			return np.stack((df_I0, df_sc, df_k), axis=1)


		#if constrained at Iref

		def aux_f3(I, I0, scale, k):

			Delta_I=np.maximum((I-I0), 0 )

			Delta_Iref=np.maximum((Iref-I0), 0 )
			return (1-np.exp( -(Delta_I/scale)**k))/(1-np.exp( -(Delta_Iref/scale)**k))

		def aux_jac3(I, I0, scale, k):
			Delta_I=np.maximum((I-I0), 0 )

			Delta_Iref=np.maximum((Iref-I0), 0 )

			xx=Delta_I/scale
			xx0=Delta_Iref/scale
			temp=np.exp( -xx**k)
			temp0=np.exp( -xx0**k)
			templ=np.log(xx+1e-6)
			templ0=np.log(xx0+1e-6)

			den=(1-temp0)**2
			df_sc=(-temp*xx**k + temp0*xx0**k + temp*temp0 * (xx**k-xx0**k))*k*1/scale*1/den

			df_k= (temp*xx**k*templ-temp0*xx0**k*templ0-temp*temp0*(templ*xx**k-templ0*xx0**k) )*1/den

			df_I0=(-temp*xx**(k-1)+temp0*xx0**(k-1)+temp*temp0*(xx**(k-1)-xx0**(k-1)))*k*1/scale*1/den
			
			return np.stack((df_I0, df_sc, df_k), axis=1)




		if init_with_new_values:
			p0 = (np.amin(I_values), np.median(I_values) - np.amin(I_values) , 2)
		else:
			p0= (self.I0.numpy(), self.scale.numpy(), self.k.numpy())



		if constrained_at_Iref:
			params, _= curve_fit(aux_f3, I_values, m_values,
				p0= p0, method='dogbox', jac=aux_jac3, ftol=0.1, 
				bounds=([-np.inf, -np.inf, 1], [np.inf, np.inf, 20]))
			self.mmax.data=1/torch.tensor( aux_f(Iref, params[0], params[1], params[2] ))
		else:
			params, _= curve_fit(aux_f, I_values, m_values,
				p0= p0, method='dogbox', jac=aux_jac, ftol=0.1, 
				bounds=([-np.inf, -np.inf, 1], [np.inf, np.inf, 20]))

		print(f'fitting data:\n I0={params[0]:.2f}, scale={params[1]:.2f}, k={params[2]:.2f}')
		self.I0.data=torch.tensor(params[0])
		self.scale.data=torch.tensor(params[1])
		self.k.data=torch.tensor(params[2])

	@classmethod
	def from_datapts(cls, I_values, m_values, requires_grad=False):
		'''
		Returns an object of the class and fits the parameter to (I_values, m_values) (see method fit_data)
		'''
		wcdf=cls(0, 10, 2, requires_grad=requires_grad)
		wcdf.fit_data(I_values, m_values)
		return wcdf

	def write_to_npz(self, filename):
		def get_data(t):
			return t.detach().numpy()

		np.savez(filename, I0=get_data(self.I0),
			scale=get_data(self.scale), k=get_data(self.k), Iref=self._Iref, mmax=get_data(self.mmax),
			constrained_at_Iref=self.constrained_at_Iref)

	@classmethod
	def load_from_npz(cls, filename, requires_grad=False):
		with np.load(filename) as f:
			I0=f['I0']
			scale=f['scale']
			k=f['k']
			mmax=f['mmax']
			Iref=float(f['Iref'])
			constrained_at_Iref=f['constrained_at_Iref']
		return cls(I0=I0, scale=scale, k=k, mmax=mmax, requires_grad=requires_grad,
			constrained_at_Iref=constrained_at_Iref, Iref=Iref)

	def __repr__(self):
		I0_str=f'{self.I0.detach().numpy():.3f}' if self.constant_I0 else '(set by RBF)'
		st=f'Weibull CDF function, I0={I0_str}, scale={self.scale.detach().numpy():.3f}, k={self.k.detach().numpy():.3f}'
		if self.constrained_at_Iref:
			st+=f' Iref:{self._Iref:.3f}'
		else:
			st+=f' mmax:{self.mmax.detach().numpy():.3f}'
		return st





def get_masking_amount(mdFunc, sq_exc, f=0., eps=1e-6):
	'''
	Args:
		mdFunc: masking degree function
		sq_exc: squared excitation associated with masking patterns
	'''
	return mdFunc(10*torch.log10(eps+sq_exc), f)




###########  Masking conditions (maskers) ###########


class MaskingConditions:
	'''
	Masking conditions (representing Gaussian noise maskers defined by bands)
	Attributes:
		n_bands: max number of bands
		n_conditions: number of masking conditions
		amp_list: torch tensors (dim: nb_conditions) of amplitudes by bands
		f_low_list: torch tensors (dim: nb_conditions) of low cut-off frequencies by bands
		f_high_list: torch tensors (dim: nb_conditions) of high cut-off frequencies by bands
		amp0: amplitude of reference (amp=1 defined in maskers means that amplitude is amp0) (default: 1). Taken in account when returning tensors
		mat_release: (mat_ref_maskers) matrix to compute the release of masking. Default:None (considering the broandband noise condition as reference). Torch tensor.
	'''

	def __init__(self, stim_dic_list=[], mat_release=None):
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
		self.mat_release=mat_release
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

	def pad_maskers(self, f_thr=11000, f_max=1e5):
		'''hack function that sets f_high frequencies above f_thr at f_max'''
		for i in range(self.n_bands):
			for j in range(len(self._f_high_list[i])):
				if self._f_high_list[i][j]>f_thr:
					self._f_high_list[i][j]=f_max

	def pad_maskers2(self, f_thr=300, f_min=0):
		'''hack function that sets f_low frequencies below f_thr at f_min'''
		for i in range(self.n_bands):
			for j in range(len(self._f_low_list[i])):
				if self._f_low_list[i][j]<f_thr:
					self._f_low_list[i][j]=f_min


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
					amp=self.amp0*self._amp_list[k][cond]
					f_low=self._f_low_list[k][cond]
					f_high=self._f_high_list[k][cond]
					if k!=0:
						st+='; '
					st+=f'({amp:.4f}, {f_low} Hz, {f_high} Hz)'
				st+=f']'
		return st







def plotMaskingDegreeFunc(maskingDegreeFunction, f=0.):
	I=torch.linspace(0, 100, 500)

	pl.figure()
	pl.plot(I, maskingDegreeFunction(I, f)*100)
	pl.xlabel('Spectral intensity (dB)')
	pl.ylabel('masking degree (%)')
	pl.ylim([0, 105])
	pl.show()



#util function to find iso-masking level curves

def aux_rec_dich(I_left, I_right, f_func, level, eps=0.001):
	'''
	Dichotomy search, assumes the function is monotonous.
	Returns:
		I such that f(I)=level
	''' 
	center=(I_left+I_right)/2
	val=f_func(center)
	err=val-level
	if np.abs(err)<eps:
		return center
	elif err<0:
		return aux_rec_dich(center, I_right, f_func, level, eps=eps)
	else:
		return aux_rec_dich(I_left, center, f_func, level, eps=eps)



def isomasking_curves_wbcdf(f_min, f_max, num_f, num_levels, I0_func, scale_func, 
	k_func, constrained_at_Iref=False, Iref=100, eps=0.001):
	'''
	Args:
		f_min:
		f_max:
		num_f: number of points between f_min and f_max
		num_levels: number of levels between 0% and 100%
		I0_func: callable for I0 as a function of CF
		scale_func: callable for scale
		k_func: callable for k
		constrained_at_Iref: passes the argument to WeibullCDF
		Iref: passes the argument to WeibullCDF. is used as upper value for the search
	Returns:
		array: f linspace
		array: level linspace
		array array: 2d array (size: num_levels, num_f) of masker PSD for each masking level

	'''
	f_arr=np.linspace(f_min, f_max, num=num_f)
	level_arr=np.linspace(0, 1, num=num_levels)
	res=np.zeros((num_levels, num_f))
	for i, f in enumerate(f_arr):
		I0=I0_func(f)
		f_func=wb_cdf=WeibullCDF_IOFunc(I0, scale_func(f), k_func(f),
		 constrained_at_Iref=constrained_at_Iref, Iref=Iref)
		I_left=I0
		I_right=Iref
		for j in range(num_levels//2):
			level_j=level_arr[j]
			I_j=aux_rec_dich(I_left, I_right, f_func, level_j, eps=eps)

			I_left=I_j

			level_mj=level_arr[num_levels-1-j]
			I_mj=aux_rec_dich(I_left, I_right, f_func, level_mj, eps=eps)
			
			I_right=I_mj

			res[j, i]=I_j
			res[num_levels-1-j, i]=I_mj
		if (num_levels%2)==1: #consider the case when j=num_levels//2 if odd
			j=num_levels//2
			level_j=level_arr[j]
			I_j=aux_rec_dich(I_left, I_right, f_func, level_j, eps=eps)

			res[j, i]=I_j

	return f_arr, level_arr, res









