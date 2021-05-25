import torch
import torch.distributed as dist

import numpy as np

import json 



class RBFNet(nn.Module):
	'''regression RBF neural net (written from Q10RBFNet see tuning.py).
	NB: diff fron Q10RBFNet is input is not log f but f directly'''
	def __init__(self, n, f_min=800, f_max=15000, sig=0.1, init_random=False):
		'''
		:param n: number of gaussians rbf (centers)
		:param init_random: if True, random initialization of centers, if False, centers are set at regular intervals
		:param sig: sigma for gaussian kernel (NB: inputs are normalized between 0 and 1)'''
		super().__init__()

		self.f_min=f_min
		self.f_max=f_max
		self.sig=sig
		self.n_centers=n
		self.sig2=sig**2
		
		def random_centers(m):
			randT=torch.rand(1, m)
			#if normalized coords used instead, not necessary :
			#randT.mul_(torch.tensor(f_max-f_min))
			#randT.add_(torch.tensor(f_min))
			return randT
		
		def init_centers(m):
			return torch.linspace(0, 1, m, requires_grad=True, dtype=torch.float)
		
		if init_random:
			init_func = random_centers
		else:
			init_func = init_centers
		self.centers = init_func(n)
		self.l2 = nn.Linear(n, 1, bias=False)
		
	def normalized_coord(self, f):
		'''NB: takes the log for f'''
		f_norm = (f-self.f_min)/(self.f_max-self.f_min)
		return f_norm
	
	def real_coord(self, f_norm):
		f = self.f_min + (self.f_max-self.f_min)*f_norm
		return f
	
	   
	def forward(self, f, verbose=False):
		f_norm = self.normalized_coord(f)
		f_norm.unsqueeze_(-1)
		sq_distances = (self.centers-f_norm)**2
		l1 = torch.exp(-sq_distances/(2*self.sig2))
		out = self.l2(l1)
		return out

	def update_weights(self,src=0, tag=17):
		'''update the weights received from another node (distributed setting). asynchronous but waiting for the update'''
		handle=dist.irecv(self.l2.weight, src=src, tag=tag)
		handle.wait()

	@classmethod
	def create_from_jsonfile(cls, filename):
		
		with open(filename) as json_file:
			data = json.load(json_file)
			n=int(data['n'])
			sig=float(data['sig'])
			f_min=float(data['f_min'])
			f_max=float(data['f_max'])
		return cls(n, f_min=f_min, f_max=f_max, sig=sig) #nb:init random always False

