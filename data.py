
import numpy as np 
import re

from scipy.io import loadmat

from masking import MaskingConditions 

#TODO class instead?

def loadMatFiles(root_folder, filenames, begin_ind=0, end_ind=np.infty):
	'''
	Args:
		begin_ind: min pic number
		end_ind: max pic number
	Returns:
		maskerNames: names of maskers in order
		maskingConditions: MaskingConditions object
		t: (numpy array)
		CAP_signals: (numpy array, of size (n_conditions, n_samples))
		map_table: mapping masker filename -> pic numbers
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
	maskingConditions=MaskingConditions()
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
					amp=pic['bands'][0][k_band]['amplitude'][0][0]
					fc_low=pic['bands'][0][k_band]['fc_low'][0][0]
					fc_high=pic['bands'][0][k_band]['fc_high'][0][0]
					stim_dic['bands'].append({'amplitude':amp, 'fc_low':fc_low, 'fc_high':fc_high})
					stim_dic['name']=pic['masker_name'][0]
				maskingConditions.add_conditions([stim_dic])
			else:
				#val+=np.sum(picDic['valAll'][1::2], axis=0)
				val+=np.squeeze(picDic['valAvg'].T)
		val/=len(picNums)
		arr_list.append(val)

	t=np.linspace(picDic['XstartPlot_ms'][0][0], picDic['XendPlot_ms'][0][0], num=len(val))
	CAP_signals=np.stack(arr_list)

	#return picDic
	return maskerNames, maskingConditions, t, CAP_signals, filtered_map_table
	
