import torch
import copy
import numpy as np
import matplotlib.pyplot as pl


from scipy.optimize import curve_fit

class PowerLawLatencies:
	'''
	Links frequencies and latencies (power law model).

	log(f)= log(A) + alpha log (t-t0) 
	Attributes:
		A
		alpha
		t0
	'''

	def __init__(self, A, alpha, t0=0., requires_grad=False, name=""):
		self.A=torch.tensor(A, requires_grad=requires_grad)
		self.alpha=torch.tensor(alpha, requires_grad=requires_grad)
		self.t0=torch.tensor(t0, requires_grad=requires_grad)
		self.name=name

	@classmethod
	def fromPts(cls, t1, f1, t2, f2, t0=0, name=""):
		det = np.log(t2-t0)-np.log(t1-t0)
		alpha=(np.log(f2)-np.log(f1))/det
		beta= (np.log(t2-t0)*np.log(f1)-np.log(t1-t0)*np.log(f2))/det
		return cls(np.exp(beta), alpha, t0, name=name)

	@classmethod
	def dilate(cls, lat, a, f_0):
		'''
		Local dilatation with fixed point at f_0
		Class method.
		Args:
			lat: PowerLawLatencies object
			a : dilatation factor
			f_0: frequency for fixed point
		Returns:
			A new PowerLawLatencies object
		'''
		with torch.no_grad():
			t_f0=lat.t_from_f(f_0)
			alpha=lat.alpha.clone()/a
			A=f_0/torch.pow(t_f0-lat.t0, alpha)
		return cls(A, alpha, t0=lat.t0)

	@classmethod
	def shift(cls, lat, t0, reinitShift=False):
		'''
		Class method.
		Args:
			lat: PowerLawLatencies object
			t0 (float) : shift in s (warning: it adds to the initial value if it applies, unless reinitShift is True) 
		Returns:
			A new PowerLawLatencies object, lat shifted by t0 
		'''
		with torch.no_grad():
			alpha=lat.alpha.clone()
			A=lat.A.clone()
			t0=(1-reinitShift)*lat.t0+t0
		return cls(A, alpha, t0=t0)
		
	def f_from_t(self, t):
		return self.A*torch.pow(t-self.t0, self.alpha)

	def t_from_f(self, f):
		return self.t0 + torch.pow(f/self.A, 1/self.alpha)

	def __call__(self, f):
		return self.t_from_f(f)

	def __repr__(self):
		return f"PowerLawLatencies obj. {self.name} \n (A={self.A.numpy():.2e}, alpha={self.alpha.numpy():.2e}, t0={self.t0.numpy():.2e})"


	def fit_data(self, t_values, f_values, init_with_new_values=True):
		'''
		Sets A, alpha and t0 to fit t_values (np array, in s) and f_values (np array).
		Levenberg-Maquardt algorithm. 
		Args:
			init_with_new_values: if True, initialization of algorithm with (0.2, -2, min t - 1 ms) , if false init with values defined by class init
		'''

		def aux_f(t, A, alpha, t0):
			return A*np.power(t-t0, alpha)

		def aux_jac(t, A, alpha, t0):
			temp = np.power(t-t0, alpha)
			df_A=temp
			df_alpha=A*temp*np.log(t-t0)
			df_t0=-A*alpha*np.power(t-t0, alpha-1)
			return np.stack((df_A, df_alpha, df_t0), axis=1)

		if init_with_new_values:
			p0=(0.2, -2, np.amin(t_values) - 1e-3)
		else:
			p0=(self.A.numpy(), self.alpha.numpy(), self.t0.numpy())

		params, _= curve_fit(aux_f, t_values, f_values,
		 	p0= p0, method='lm', jac=aux_jac)


		print(f'fitting data:\n A={params[0]:.3f}, alpha={params[1]:.2f}, t0={params[2]*1e3:.2f} ms')

		self.A.data=torch.tensor(params[0])
		self.alpha.data=torch.tensor(params[1])
		self.t0.data=torch.tensor(params[2])




Eggermont1976clickLatencies80dB=PowerLawLatencies.fromPts(5.3e-3, 1e3, 2e-3, 5e3, name="Eggermont 1976 click 80dB")


def plotLatencies(lat):
	t0=lat.t0

	tmin=lat.t_from_f(torch.tensor(10e3))

	t=torch.linspace(tmin-t0, 10e-3-t0, 200) #t - t0
	f=lat.f_from_t(t+t0)

	pl.figure()
	pl.title(f'Latencies {lat.name}')
	pl.plot(t*1e3, f)
	pl.xlabel('t-t0 (ms)')
	pl.ylabel('f (Hz)')
	pl.xscale('log')
	pl.yscale('log')
	pl.show()

	pl.figure()
	pl.title(f'Latencies {lat.name}')
	pl.plot((t0+t)*1e3, f*1e-3)
	pl.xlabel('t (ms)')
	pl.xlim([0, 10])
	pl.ylabel('f (kHz)')
	pl.show()

