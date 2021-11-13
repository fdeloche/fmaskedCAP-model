 
import numpy as np 
import re

from scipy.io import loadmat

from masking import MaskingConditions 

from functools import partial

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


	def __init__(self, root_folder, filenames, begin_ind=0, end_ind=np.infty, old_format=False, mode='C+R', 
		pic_numbers_ignore=[]):
		'''
		Args:
			root_folder: root folder for .mat files
			filenames: list of *.mat files with CAPs and info on maskers
			begin_ind: min pic number
			end_ind: max pic number
			old_format:old format of .mat files
			mode: 'C+R', 'C' 'R' (check C and R? #TODO )
			pic_numbers_ignore: list of pic numbers to ignore
		'''  

		assert mode in ['C+R', 'C', 'R']

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

			m=re.match('p([0-9]{4})_fmasked_CAP_(.*).mat', fln)
			if not m:
				print(f'warning: {fln} did not match reg filter')
				continue

			p, name = split_name(fln)
			if p>=begin_ind and p<=end_ind and not (p in pic_numbers_ignore):

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
			if old_format:
				return arr
			else:
				return arr['data_struct']

				
		def get_info_pic(pic, mode='C+R'):
			'''
			Args:
				mode: 'C+R', 'C' 'R' (check C and R? #TODO )
			'''

			if mode=='C+R':
				#avg
				arr=pic['AD_Data'][0][0]['AD_Avg_V'][0][0][0]
				
			if mode=='C' or mode=='R':
				rem= 0 if mode == 'C' else 1
				all_data=pic['AD_Data'][0][0]['AD_All_V'][0][0]
				arr=np.zeros_like(all_data[0])
				for i in range(len(all_data)):
					if i%2==rem:
						arr+=all_data[i]
				arr/=len(all_data)/2


			n_bands=pic['Stimuli'][0][0]['masker'][0][0]['n_bands'][0][0][0][0]
			amps=pic['Stimuli'][0][0]['masker'][0][0]['bands'][0][0]['amplitude'] if n_bands>0 else []
			fcs_low=pic['Stimuli'][0][0]['masker'][0][0]['bands'][0][0]['fc_low'] if n_bands>0 else []
			fcs_high=pic['Stimuli'][0][0]['masker'][0][0]['bands'][0][0]['fc_high'] if n_bands>0 else []

			return {'arr':arr,
					'XstartPlot_ms':pic['Stimuli'][0][0]['CAP_intervals'][0][0]['XstartPlot_ms'][0][0][0],
					'XendPlot_ms': pic['Stimuli'][0][0]['CAP_intervals'][0][0]['XendPlot_ms'][0][0][0],
					'name':pic['Stimuli'][0][0]['masker'][0][0]['name'][0][0],
					'n_bands':n_bands,
					'amps': amps,
					'fcs_low': fcs_low,
					'fcs_high': fcs_high}


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
					if old_format:
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
						firstPic=False
					else:
						pic_info=get_info_pic(picDic, mode=mode)
						val=pic_info['arr']
						#info on masker
						pic=pic_info
						stim_dic={}
						stim_dic['n_bands']=n_bands=pic['n_bands']
						stim_dic['bands']=[]
						for k_band in range(n_bands):
							amp=float(pic['amps'][k_band])
							fc_low=float(pic['fcs_low'][k_band])
							fc_high=float(pic['fcs_high'][k_band])
							stim_dic['bands'].append({'amplitude':amp, 'fc_low':fc_low, 'fc_high':fc_high})
							stim_dic['name']=pic['name']
						list_stim_dic.append(stim_dic)
						firstPic=False

				else:
					if old_format:
						val+=np.squeeze(picDic['valAvg'].T)
					else:

						pic_info=get_info_pic(picDic, mode=mode)
						val+=pic_info['arr']
			val/=len(picNums)
			arr_list.append(val)

		t=np.linspace(pic['XstartPlot_ms'], pic['XendPlot_ms'], num=len(val))*1e-3
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

	def get_signal_by_name(self, maskerName):
		ind=self.maskerNames.index(maskerName)
		#print(self.map_table[maskerName])
		return self.CAP_signals[ind]
	
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


	def get_batch_re(self, reg_expr):
		'''return a batch (maskerNames, maskingConditions, CAPsignals) with maskers corresponding to a regular expression'''
		inds=[]
		for ind, maskerName in enumerate(self.maskerNames):
			if re.match(reg_expr, maskerName):
				inds.append(ind)
		inds=sorted(inds, key= lambda ind: self.maskerNames[ind]) 
		obj=self
		batch = ([obj.maskerNames[ind] for ind in inds], 
				MaskingConditions([obj.list_stim_dic[ind] for ind in inds]),  obj.CAP_signals[inds])
		return batch





