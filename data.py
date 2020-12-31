
import numpy as np 
import re

from scipy.io import loadmat

from masking import MaskingConditions 

from functools import partial
#TODO class instead?

class CAPData:
	'''
	Attributes:
		maskerNames: list masker names
		nb_maskers
		list_stim_dic: list stim_dic (info on maskers)
		maskingConditions: maskingConditions object corresponding to list_stim_dict (created at first call)
		CAP_signals: (numpy array, of size (n_conditions, n_samples))
		t: numpy array, in s
		map_table: mapping masker filename -> pic numbers
	'''


	def __init__(self, root_folder, filenames, begin_ind=0, end_ind=np.infty):
		'''
		Args:
			root_folder: root folder for .mat files
			filenames: list of *.mat files with CAPs and info on maskers
			begin_ind: min pic number
			end_ind: max pic number
		'''  

		listFilesMat=filenames
		data_folder=root_folder

		def split_name(filename):
			'''returns (picNumber, masker filename)'''
			m=re.match('p([0-9]{4})_fmasked_CAP_(.*).mat', filename)
			assert m, f'{filename} dit not match'
			p, name= m.groups()
			return int(p), name


		filtered_map_table={}  #map masker filename -> pic numbers
		picnum_to_filename={} #secondary map table
		for fln in listFilesMat:
			p, name = split_name(fln)
			if p>=begin_ind and p<=end_ind:

				if name not in filtered_map_table:
					filtered_map_table[name]=[p]
				else:
					li=filtered_map_table[name]
					li.append(p)
					filtered_map_table[name]=li
			picnum_to_filename[p]=fln


		def loadPic(n):
			'''found=False

			for filename in listFilesMat:
				if f'p{str(n).zfill(4)}' in filename:
					found=True
					break
			assert found, f'pic {n} not found.'
			'''
			filename=picnum_to_filename[n]
			arr= loadmat(f'{data_folder}/{filename}')
			return arr

		maskerNames=list(filtered_map_table.keys())
		#maskingConditions=MaskingConditions()
		list_stim_dic=[]
		arr_list=[]
		for maskerName in maskerNames:
			picNums=filtered_map_table[maskerName]
			firstPic=True
			for picNum in picNums:
				picDic=loadPic(picNum)
				if firstPic:
					#val=np.sum(picDic['valAll'][1::2], axis=0)
					val=np.squeeze(picDic['valAvg'].T)
					firstPic=False
					#info on masker
					pic=picDic
					stim_dic={}
					stim_dic['n_bands']=n_bands=pic['n_bands'][0][0]
					stim_dic['bands']=[]
					for k_band in range(n_bands):
						amp=float(pic['bands'][0][k_band]['amplitude'][0][0])
						fc_low=float(pic['bands'][0][k_band]['fc_low'][0][0])
						fc_high=float(pic['bands'][0][k_band]['fc_high'][0][0])
						stim_dic['bands'].append({'amplitude':amp, 'fc_low':fc_low, 'fc_high':fc_high})
						stim_dic['name']=pic['masker_name'][0]
					list_stim_dic.append(stim_dic)
					#maskingConditions.add_conditions([stim_dic])
				else:
					#val+=np.sum(picDic['valAll'][1::2], axis=0)
					val+=np.squeeze(picDic['valAvg'].T)
			val/=len(picNums)
			arr_list.append(val)

		t=np.linspace(picDic['XstartPlot_ms'][0][0], picDic['XendPlot_ms'][0][0], num=len(val))*1e-3
		CAP_signals=np.stack(arr_list)

		self.t=t
		self.list_stim_dic=list_stim_dic
		self.CAP_signals=CAP_signals
		self.maskerNames=maskerNames
		self.map_table=filtered_map_table
		self.nb_maskers=len(self.maskerNames)

	@property
	def maskingConditions(self):
		#similar to cached_property
		if not hasattr(self, '_maskingConditions'):
			self._maskingConditions=MaskingConditions(stim_dic_list=self.list_stim_dic)
		return self._maskingConditions
	
	def batch_generator(self, batch_size):
		'''return batches of size batch_size with tuples (maskerNames, maskingConditions, CAPsignals)'''
		list_ind=np.random.permutation(self.nb_maskers)
		s=self.nb_maskers-(self.nb_maskers%batch_size)
		list_ind=list_ind[0:s]
		list_ind_batches=np.reshape(list_ind, (s//batch_size, batch_size))
		obj=self
		for indices in list_ind_batches:
			batch = ([obj.maskerNames[ind] for ind in indices], 
				MaskingConditions([obj.list_stim_dic[ind] for ind in indices]),  obj.CAP_signals[indices])
			yield batch




