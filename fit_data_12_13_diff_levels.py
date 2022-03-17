CF=5000 #HACK for now

#Note: list maskers at end of file

import config_mode
config_mode.mode='C+R'

import torch

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pl
import scipy.signal as sg


from scipy.ndimage  import gaussian_filter1d

import json
import re
import os

#import copy

from masking import *
from latencies import *
from excitation import *
from deconv import *
from ur import *
from tuning import *
from test import *
from ur import *

from data import CAPData



data_folder='./Data/AS-2021_12_13-ChinQ336_fmaskedCAP_normal'

fs=48828

I0 = 106 #intensity ref for masker    #105.62 dB rms=1   #For 0 dB atten  #NO +11 dB (previous mistake)
I0 = 10*np.log10( 10**(I0/10)/(fs/2) ) #spectral density


assert CF==5000, 'CF must be 5kHz (only data at that CF'


print(f'reference masker power spectral density (0 attn): {I0:.2f} dB')


import_results_folder='./results/fit12-13-021122-run5/'
import_ur=True 
import_weights=True

### capData


listFiles = os.listdir(data_folder)

assert config_mode.mode is not None, 'cap mode (config_mode.mode) not set'

capMode=config_mode.mode

dic_ref_maskers={
 '1_hp6000_narrowband5kHz_45dB':  '8_hp6000_narrowband5kHz_20dB',
 '2_hp6000_narrowband5kHz_40dB':  '8_hp6000_narrowband5kHz_20dB',
 '3_hp6000_narrowband5kHz_35dB':  '8_hp6000_narrowband5kHz_20dB',
 '4_hp6000_narrowband5kHz_32dB':  '8_hp6000_narrowband5kHz_20dB',
 '5_hp6000_narrowband5kHz_29dB':  '8_hp6000_narrowband5kHz_20dB',
 '6_hp6000_narrowband5kHz_26dB':  '8_hp6000_narrowband5kHz_20dB',
 '7_hp6000_narrowband5kHz_23dB':  '8_hp6000_narrowband5kHz_20dB',
 '8_hp6000_narrowband5kHz_20dB':  '8_hp6000_narrowband5kHz_20dB',
 '9_hp6000_narrowband5kHz_17dB':  '8_hp6000_narrowband5kHz_20dB',
 '10_hp6000_narrowband5kHz_14dB':  '8_hp6000_narrowband5kHz_20dB',
 '11_hp6200_gradualamp':  '15_hp6200_gradualamp',
 '12_hp6200_gradualamp':  '15_hp6200_gradualamp',
 '13_hp6200_gradualamp':  '15_hp6200_gradualamp',
 '14_hp6200_gradualamp':  '15_hp6200_gradualamp',
 '15_hp6200_gradualamp':  '15_hp6200_gradualamp',
 '16_hp6200_gradualamp':  '15_hp6200_gradualamp',
 '17_notch5300_bw1000': '21_notch5300_bw1800_nonotch',
 '18_notch5300_bw1200': '21_notch5300_bw1800_nonotch',
 '19_notch5300_bw1400': '21_notch5300_bw1800_nonotch',
  '20_notch5300_bw1800_attn24': '21_notch5300_bw1800_nonotch',
 '21_notch5300_bw1800_nonotch': '21_notch5300_bw1800_nonotch',
 '22_notch5k': '21_notch5300_bw1800_nonotch',
 '23_notch4800_bw900': '21_notch5300_bw1800_nonotch',
 '24_notch4800_bw800': '21_notch5300_bw1800_nonotch',
 }



# indices
dic_inds={
 40: [(1487, 1534), (1782, 1829), (2070, 2117), (2358, 2405), (2646, 2693)] ,
 35: [(1535, 1582), (1830, 1877), (2118, 2165), (2406, 2453), (2694, 2741)],
 30: [(1583, 1630), (1878, 1925), (2166, 2213), (2454, 2501), (2742, 2789)],
 25: [(1631, 1678), (1926, 1973), (2214, 2261), (2502, 2549), (2790, 2837)],
 20: [(1686, 1733), (1974, 2021), (2262, 2309), (2550, 2597), (2838, 2885)],
 15: [(1734, 1781), (2022, 2069), (2310, 2357), (2598, 2645), (2886, 2933)]
 }

capDataDic={}  #dic atten->capData
for atten, list_tuples in dic_inds.items():
	begin_inds, end_inds=zip(*list_tuples)
	capData=CAPData(data_folder, listFiles, begin_ind=begin_inds, end_ind=end_inds, 
		mode=capMode, pic_numbers_ignore=[], verbose=False)
	capData.set_ref_maskers(dic_ref_maskers)
	capDataDic[atten]=capData

t=capData.t



sig_ex=capData.get_signal_by_name('3_hp6000_narrowband5kHz_35dB')


def plot_main_CAPs(**kwargs):
	pl.figure(**kwargs)
	pl.plot(t*1e3, sig_ex*1e3, label='3_hp6000_narrowband5kHz_35dB')

	sig_ex2=capData.get_signal_by_name('8_hp6000_narrowband5kHz_20dB')
	pl.plot(t*1e3, sig_ex2*1e3, label='8_hp6000_narrowband5kHz_20dB')

	pl.xlabel('t (ms)')
	pl.ylabel('Amplitude (μV)')
	#pl.xlim([0.004, 0.007])
	pl.legend()
	pl.show()



### Windowing/processing
#NB: 1st processing (truncating), 
# 2nd processing (diff with broadband condition + smoothing) + windowing

t0=5.7e-3
t1=9.4e-3  #previously: 10e-3  (last part not very reliable)
ind0=int(t0*48828)

ind0=int(t0*48828)
ind1=int(t1*48828)

alpha_tukey=0.4
win0=sg.tukey(ind1-ind0, alpha=alpha_tukey)  #NB: same tukey window defined later for 2nd processing (truncated version)

win=np.zeros_like(sig_ex)
win[ind0:ind1]=win0




def plot_CAP_with_window(**kwargs):
	pl.figure(**kwargs)
	sig=capData.get_signal_by_name('3_hp6000_narrowband5kHz_35dB')-capData.get_signal_by_name('8_hp6000_narrowband5kHz_20dB')
	pl.plot(t*1e3, sig*1e3, label='hp6000_narrowband5kHz 35 dB atten (- ref: 20 dB)')
	pl.plot(t*1e3, win*np.amax(sig)*1e3)

	pl.plot(t*1e3, sig*win*1e3, label='avg windowed')

	pl.xlabel('t (ms)')

	pl.ylabel('Amplitude difference (μV)')
	pl.legend()
	pl.show()


def process_signal(sig,  return_t=False):
	#sig2=sig*win  #done in process2
	sig2=sig

	t0=3e-3
	t1=13e-3

	ind0=int(t0*48828)
	ind1=int(t1*48828)
	
	dim = len(np.shape(sig2))
	if dim ==1:
		sig2=sig2[ind0:ind1]

	else:
		sig2=sig2[:, ind0:ind1]		
	
	if return_t:
		t=np.linspace(t0, t1, ind1-ind0)
		return t, sig2
	else:
		return sig2
	

t2, sig_ex_proc=process_signal(sig_ex, return_t=True)

dt=t2[1]-t2[0]

t0=t0-3e-3
t1=t1-3e-3

ind0=int(t0*48828)
ind1=int(t1*48828)

win20=sg.tukey(ind1-ind0, alpha=alpha_tukey)

win2=np.zeros_like(sig_ex_proc)
win2[ind0:ind1]=win20


def process_signal2(sig, gauss_sigma=0, corr_drift=True):
	'''
	Note: before, the response corresponding to the broadband noise masker was subtracted. Now substraction is done through multiplication with matrix build with ref maskers (outside function, see CAPData in data.py)

	gauss_sigma: if diff of 0, smooths the signal with gaussian filter'''
	
	res = process_signal(sig)


	if gauss_sigma !=0:
		res = gaussian_filter1d(res, gauss_sigma)

	res*=win2

	return res



gauss_sigma=(0.3e-4)/(t2[1]-t2[0]) #01/19/22

# ur

data_ur=np.load(f'{import_results_folder}/ur_{CF}.npz')
ur0, t_conv= data_ur['ur'], data_ur['t2']

# latencies

lat=PowerLawLatencies.load_from_npz(f'{import_results_folder}/lat_{CF}.npz')

# weights

data_weights= np.load(f'{import_results_folder}/E0_{CF}.npz')
weights_f, weights_E0, weights_E0_amp=data_weights['f'], data_weights['E0'], data_weights['E0_amp']

#Q10?



#### List maskers


# dic_ref_maskers={
#  '1_hp6000_narrowband5kHz_45dB':  '8_hp6000_narrowband5kHz_20dB',
#  '2_hp6000_narrowband5kHz_40dB':  '8_hp6000_narrowband5kHz_20dB',
#  '3_hp6000_narrowband5kHz_35dB':  '8_hp6000_narrowband5kHz_20dB',
#  '4_hp6000_narrowband5kHz_32dB':  '8_hp6000_narrowband5kHz_20dB',
#  '5_hp6000_narrowband5kHz_29dB':  '8_hp6000_narrowband5kHz_20dB',
#  '6_hp6000_narrowband5kHz_26dB':  '8_hp6000_narrowband5kHz_20dB',
#  '7_hp6000_narrowband5kHz_23dB':  '8_hp6000_narrowband5kHz_20dB',
#  '8_hp6000_narrowband5kHz_20dB':  '8_hp6000_narrowband5kHz_20dB',
#  '9_hp6000_narrowband5kHz_17dB':  '8_hp6000_narrowband5kHz_20dB',
#  '10_hp6000_narrowband5kHz_14dB':  '8_hp6000_narrowband5kHz_20dB',
#  '11_hp6200_gradualamp':  '15_hp6200_gradualamp',
#  '12_hp6200_gradualamp':  '15_hp6200_gradualamp',
#  '13_hp6200_gradualamp':  '15_hp6200_gradualamp',
#  '14_hp6200_gradualamp':  '15_hp6200_gradualamp',
#  '15_hp6200_gradualamp':  '15_hp6200_gradualamp',
#  '16_hp6200_gradualamp':  '15_hp6200_gradualamp',
#  '17_notch5300_bw1000': '21_notch5300_bw1800_nonotch',
#  '18_notch5300_bw1200': '21_notch5300_bw1800_nonotch',
#  '19_notch5300_bw1400': '21_notch5300_bw1800_nonotch',
#   '20_notch5300_bw1800_attn24': '21_notch5300_bw1800_nonotch',
#  '21_notch5300_bw1800_nonotch': '21_notch5300_bw1800_nonotch',
#  '22_notch5k': '21_notch5300_bw1800_nonotch',
#  '23_notch4800_bw900': '21_notch5300_bw1800_nonotch',
#  '24_notch4800_bw800': '21_notch5300_bw1800_nonotch',
#  }

#follows same format as maskers for previous expes for consistency (even if names are not very accurate)
# caution: also include ref maskers in masker lists

ntch_5k_masker_list=['1_hp6000_narrowband5kHz_45dB',
'2_hp6000_narrowband5kHz_40dB',
'3_hp6000_narrowband5kHz_35dB',
'4_hp6000_narrowband5kHz_32dB',
'5_hp6000_narrowband5kHz_29dB',
'6_hp6000_narrowband5kHz_26dB',
'7_hp6000_narrowband5kHz_23dB',
'8_hp6000_narrowband5kHz_20dB',
'9_hp6000_narrowband5kHz_17dB',
'10_hp6000_narrowband5kHz_14dB']

ntch_5k_masker_list+=['15_hp6200_gradualamp', '16_hp6200_gradualamp'] #extra maskers

ntch_5k_re='\d*_hp6000_narrowband5kHz_\d*dB'

vbw_5k_fln_list=['17_notch5300_bw1000',
  '18_notch5300_bw1200',
  '19_notch5300_bw1400',
  '21_notch5300_bw1800_nonotch',
  '22_notch5k',
  '23_notch4800_bw900',
  '24_notch4800_bw800']

vbw_5k_fln_list +=['11_hp6200_gradualamp',
  '12_hp6200_gradualamp',
 '13_hp6200_gradualamp',
  '14_hp6200_gradualamp',
  '15_hp6200_gradualamp',
  '16_hp6200_gradualamp']


vfreq_fln_lists={ 5000:[]}  #import weights instead     


vbw_fln_lists={5000:vbw_5k_fln_list}

attns_5k=np.array([45,40,35,32,29,26,23, 17, 14])

ntch_regexps={5000:ntch_5k_re}   

ntch_masker_lists={5000: ntch_5k_masker_list}

attns_arrays={
5000:attns_5k}


#more explicit labels for hp6200 gradual amp  #to be used later?
labels_gradual_amp={
 '11_hp6200_gradualamp':  '11_hp6200_gradualamp_bw1000',
 '12_hp6200_gradualamp':  '12_hp6200_gradualamp_bw700',
 '13_hp6200_gradualamp':  '13_hp6200_gradualamp_bw1400',
 '14_hp6200_gradualamp':  '14_hp6200_gradualamp_bw1200',
 '15_hp6200_gradualamp':  '15_hp6200_gradualamp_nonotch',
 '16_hp6200_gradualamp':  '16_hp6200_gradualamp_nonotch_27dB',
}



### Test masking releases/ ref maskers
# sig=capData.get_signal_by_name('3_hp6000_narrowband5kHz_35dB')

# sig2=capData.get_signal_by_name('8_hp6000_narrowband5kHz_20dB')

# sig3=capData.get_signal_by_name('3_hp6000_narrowband5kHz_35dB', subtract_ref=True)

# reg_exp='(3_hp6000_narrowband5kHz_35dB)|(8_hp6000_narrowband5kHz_20dB)'
# test_maskerNames, test_maskingConds, test_signals =capData.get_batch_re(reg_exp, subtract_ref=True)

# index0=test_maskerNames.index('3_hp6000_narrowband5kHz_35dB')
# index1=test_maskerNames.index('8_hp6000_narrowband5kHz_20dB')

# sig4=test_signals[index0]

# sig5=test_signals[index1]

# pl.plot(t, sig-sig2)
# pl.plot(t, sig3+1e-4)

# pl.plot(t, sig4+5e-4)

# #pl.plot(t, sig5+10e-4)