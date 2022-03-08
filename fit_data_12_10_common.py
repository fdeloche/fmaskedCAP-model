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

import config_mode

### Import data

data_folder='./Data/AS-2021_12_10-ChinQ333_CAP_normal'

fs=48828

#note: mult factor click -4.12 dB, click atten 23 dB

I0 = 102 - 35 #intensity ref for masker    #105.62 dB rms=1   #-30 dB masker atten   #NO +11 dB amp 5   (/sqrt(2))
I0 = 10*np.log10( 10**(I0/10)/(fs/2) ) #spectral density


print(f'reference masker power spectral density (0 attn): {I0:.2f} dB')

listFiles = os.listdir(data_folder)

cumsum_default=False


assert config_mode.mode is not None, 'cap mode (config_mode.mode) not set'

capMode=config_mode.mode

capData=CAPData(data_folder, listFiles, begin_ind=7, end_ind=1638, \
	mode=capMode, pic_numbers_ignore=[], verbose=False)

### Main signals


t=capData.t

broadband1=capData.get_signal_by_name('broadband_noise')
broadband2=capData.get_signal_by_name('broadband_noise_bis')
broadband3=capData.get_signal_by_name('broadband_noise_bis_bis')

broadband_avg=(broadband1+broadband2+broadband3)/3

nomasker1=capData.get_signal_by_name('nomasker')
nomasker2=capData.get_signal_by_name('nomasker_bis')

nomasker_avg=nomasker1+nomasker2
nomasker_avg/=2

def plot_main_CAPs(**kwargs):
	pl.figure(**kwargs)
	pl.plot(t*1e3, broadband1*1e3)
	pl.plot(t*1e3, broadband2*1e3)
	pl.plot(t*1e3, broadband3*1e3)
	pl.plot(t*1e3, broadband_avg*1e3, label='avg broadband')


	pl.plot(t*1e3, nomasker1*1e3, label='no masker')


	pl.plot(t*1e3, nomasker2*1e3, label='no masker bis')

	pl.xlabel('t (ms)')
	pl.ylabel('Amplitude (μV)')
	#pl.xlim([0.004, 0.007])
	pl.legend()
	pl.show()


### Windowing/processing
#NB: 1st processing (filtering), 
# 2nd processing (diff with broadband condition + smoothing) + windowing

t0=6.0e-3
t1=9.7e-3  #previously: 10e-3  (last part not very reliable)
ind0=int(t0*48828)

ind0=int(t0*48828)
ind1=int(t1*48828)

alpha_tukey=0.4
win0=sg.tukey(ind1-ind0, alpha=alpha_tukey)  #NB: same tukey window defined later for 2nd processing (truncated version)

win=np.zeros_like(broadband_avg)
win[ind0:ind1]=win0


def plot_CAP_with_window(**kwargs):
	pl.figure(**kwargs)
	sig=(nomasker_avg-broadband_avg)
	pl.plot(t*1e3, sig*1e3, label='avg (- broadband cond)')
	pl.plot(t*1e3, win*np.amax(sig)*1e3)

	pl.plot(t*1e3, sig*win*1e3, label='avg windowed')

	pl.xlabel('t (ms)')

	pl.ylabel('Amplitude difference (μV)')
	pl.legend()
	pl.show()


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
	

t2, broadband_proc=process_signal(broadband_avg, cumsum=cumsum_default, return_t=True)
nomasker_proc=process_signal(nomasker_avg)

dt=t2[1]-t2[0]

t0=t0-3e-3
t1=t1-3e-3

ind0=int(t0*48828)
ind1=int(t1*48828)

win20=sg.tukey(ind1-ind0, alpha=alpha_tukey)

win2=np.zeros_like(broadband_proc)
win2[ind0:ind1]=win20


def process_signal2(sig, cumsum=cumsum_default, gauss_sigma=0, corr_drift=True):
	'''subtracts the broadband noise response
	gauss_sigma: if diff of 0, smooths the signal with gaussian filter'''
	
	res = process_signal(sig-broadband_avg, cumsum=cumsum)


	if gauss_sigma !=0:
		res = gaussian_filter1d(res, gauss_sigma)

	res*=win2

	return res

### Estimation ur / raw excitation pattern

### XXX depends on what is the focus (CF dependent)

#sig=capData.get_signal_by_name('7_notch8000_bw2300_29dB')  #high freq
#sig=capData.get_signal_by_name('8_notch6000_bw2000_29dB')
#sig=capData.get_signal_by_name('8_notch4000_bw1700_29dB')
#sig=capData.get_signal_by_name('9_notch3000_bw1500_29dB')
#sig=capData.get_signal_by_name('8_notch2200_bw1500_29dB') #medium freq


ur0_masker_name='8_notch6000_bw2000_29dB'
sig=capData.get_signal_by_name(ur0_masker_name)


sig2=process_signal2(sig)

#ur0=sig2-broadband_proc

gauss_sigma=(0.3e-4)/(t2[1]-t2[0]) #01/19/22

ur0=process_signal2(sig, gauss_sigma=gauss_sigma)
ur0=np.roll(ur0, -50)   #100 ->50

def deconv(released_sig, ur0=ur0, eps=1e-2):
	
	released_sig_fft=np.fft.rfft(released_sig)
	ur0_fft=np.fft.rfft(ur0)
	E_fft=released_sig_fft/(ur0_fft+eps)
	E=np.fft.irfft(E_fft)
	return E
#masked_sig=nomasker_proc-broadband_proc
masked_sig=process_signal2(nomasker_avg, gauss_sigma=gauss_sigma)
E0=deconv(masked_sig)

#estimation with projection

def proj_E(E, t0=3.5e-3, t1=6.5e-3):
	'''
	constraints u between t0 and t1'''
	proj=t2>t0
	proj*=t2<t1
	return E*proj

def deconv_newton(E0, released_sig, ur0=ur0, alpha=0.02, nb_steps=20, eps_ridge=1e-1, 
	verbose=False, t0=3.5e-3, t1=6.5e-3):
	E=proj_E(E0, t0=t0, t1=t1)

	released_sig_fft=np.fft.rfft(released_sig)
	ur0_fft=np.fft.rfft(ur0)

	E=np.expand_dims(E, axis=0)

	for i in range(nb_steps):
		E-=alpha*deconv_newton_step(E, ur0_fft, released_sig_fft, eps_ridge=eps_ridge)

		E=proj_E(E, t0=t0, t1=t1)
		E[E<0]=0
		if verbose and i%5==0:
			pl.plot(t2*1e3, E[0], label=f'step {i}')
			pl.xlabel('t (ms)')
	if verbose:
		pl.legend()
	return E[0]

E=deconv_newton(E0, masked_sig, verbose=False)

def plot_raw_excitation_deconv():
	pl.figure()
	pl.title(f'ur0 ({ur0_masker_name})')
	pl.plot(t2-t2[0], ur0)
	pl.show()

	pl.figure()
	pl.plot(t2*1e3, E0, label=f'E0 (simple deconv)')
	pl.plot(t2*1e3, E, label=f'E0 (w/ proj)')
	pl.xlabel('t (ms)')
	pl.legend()
	pl.show()


### Narrowband analysis  --> not possible, forgot to collect this data

def plot_figures_narrowband_analysis():
	print('narrowband analysis method not possible, no data')


def plot_figures_narrowband_analysis_deconv():

	print('narrowband analysis method not possible, no data')




### Latencies
 
#based on notched-noise maskers instead of hp noise maskers
t_max=np.array([36.5, 38.5, 39, 39.5, 41.5])
t_max[0]=38 #HACK correct outlier
t_max=t_max/16.7

t_0lat=5e-3-4.8e-3  #ref click

t_max=t_0lat+t_max*1e-3

#t_max_bis=t_0lat+t_max_bis*2*1e-5
freqs=np.array([6.,5 ,4, 3, 2.2])  #note: no signal 8khz and 1.5 kHz


def plot_estimated_latencies_deconv():
	pl.figure()
	pl.plot(freqs, t_max*1e3, '+', markersize=12, label='C+R')

	#pl.plot(freqs[0:len(t_max_bis)], t_max_bis*1e3, '+', markersize=12, label='C+R (first peak?)')

	pl.ylabel('Estimated latencies (ms)')
	pl.xlabel('freq (kHz)')

	pl.legend()
	pl.show()
#NB click at 4.8 ms approx
#peak convol begins at 3 ms (nb: previously, 2ms)


# fit latencies power law

freqs_pts0=freqs_pts=freqs*1e3
t_max_pts0=t_max_pts=t_max

#inds=np.array([0,3,5,6,7,8,10,11]) #HACK remove 'outliers'
#freqs_pts=freqs_pts[inds]
#freqs_pts0=freqs_pts
#t_max_pts=t_max[inds]
#t_max_pts0=t_max_pts


# lat=PowerLawLatencies(1e6, alpha=1, t0=4e-3, mode='left')
# lat.fit_data(t_max_pts, freqs_pts, init_with_new_values=False, bounds=[0.5, 2])

# lat_above4k=lat


# #below 4kHz
# freqs_pts=freqs[6:]*1e3
# t_max_pts=t_max[6:]


lat=PowerLawLatencies()
lat.fit_data(t_max_pts, freqs_pts)

def plot_latencies_fit():

	pl.figure()
	freqs_lin=np.linspace(0.5, 10)*1e3
	pl.plot(freqs_lin, lat(freqs_lin)*1e3)

	pl.plot(freqs_pts, t_max_pts*1e3, '+', markeredgewidth=3, markersize=10)

	pl.show()

	pl.figure()

	pl.plot(freqs_lin, (lat(freqs_lin)-lat.t0)*1e3, color='C2', linestyle='--')

	pl.plot(freqs_pts, (t_max_pts-lat.t0.numpy())*1e3, '+', markeredgewidth=3, markersize=10)

	pl.ylabel(' t - t_0 (ms)')

	pl.xlabel(' f (Hz)')

	pl.xscale('log')
	pl.yscale('log')
	pl.show()

