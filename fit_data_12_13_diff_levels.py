CF=5000 #HACK for now
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


#more explicit labels for hp6200 gradual amp  #to be used later?
labels_gradual_amp={
 '11_hp6200_gradualamp':  '11_hp6200_gradualamp_bw1000',
 '12_hp6200_gradualamp':  '12_hp6200_gradualamp_bw700',
 '13_hp6200_gradualamp':  '13_hp6200_gradualamp_bw1400',
 '14_hp6200_gradualamp':  '14_hp6200_gradualamp_bw1200',
 '15_hp6200_gradualamp':  '15_hp6200_gradualamp_nonotch',
 '16_hp6200_gradualamp':  '16_hp6200_gradualamp_nonotch_27dB',
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

### Windowing/processing
#NB: 1st processing (filtering), 
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

cumsum_default=False

def process_signal(sig, cumsum=cumsum_default, return_t=False):
	#sig2=sig*win  #done in process2
	sig2=sig

	t0=3e-3
	t1=13e-3

	ind0=int(t0*48828)
	ind1=int(t1*48828)
	
	dim = len(np.shape(sig2))
	if dim ==1:
		sig2=sig2[ind0:ind1]
		if cumsum:
			sig2=np.cumsum(sig2)
			sig2[0:-50]*=sg.tukey(len(sig2)-50, 0.3)
			sig2[-50:]=0
	else:
		sig2=sig2[:, ind0:ind1]
		if cumsum:
			sig2=np.cumsum(sig2)
			
	
	if return_t:
		t=np.linspace(t0, t1, ind1-ind0)
		return t, sig2
	else:
		return sig2
	

t2, sig_ex_proc=process_signal(sig_ex, cumsum=cumsum_default, return_t=True)

dt=t2[1]-t2[0]

t0=t0-3e-3
t1=t1-3e-3

ind0=int(t0*48828)
ind1=int(t1*48828)

win20=sg.tukey(ind1-ind0, alpha=alpha_tukey)

win2=np.zeros_like(broadband_proc)
win2[ind0:ind1]=win20


def process_signal2(sig, cumsum=cumsum_default, gauss_sigma=0, corr_drift=True):
	'''
	Note: before, the response corresponding to the broadband noise masker was subtracted. Now substraction should be done through of matrix with ref maskers

	gauss_sigma: if diff of 0, smooths the signal with gaussian filter'''
	
	res = process_signal(sig, cumsum=cumsum)


	if gauss_sigma !=0:
		res = gaussian_filter1d(res, gauss_sigma)

	res*=win2

	return res



# latencies

# weights

# ur

#Q10?