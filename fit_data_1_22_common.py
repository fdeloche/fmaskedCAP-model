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

from Fit_data_1_22_aux import apply_filter

from masking import *
from latencies import *
from excitation import *
from deconv import *
from ur import *
from tuning import *
from test import *
from ur import *

import config_mode

from data import CAPData

### Import data

data_folder='./Data/SP-2021_01_22_Q395_fmaskedCAP'

fs=48828

I0 = 100 +10 - 32 #intensity ref for masker    #100 dB rms=1  +10 dB amp 5   (/sqrt(2))   #-32 dB masker atten
I0 = 10*np.log10( 10**(I0/10)/(fs/2) ) #spectral density

print(f'reference masker power spectral density (0 attn): {I0:.2f} dB')

listFiles = os.listdir(data_folder)

cumsum_default=False

assert config_mode.mode is not None, 'cap mode (config_mode.mode) not set'

capMode=config_mode.mode

capData=CAPData(data_folder, listFiles, begin_ind=50, end_ind=1525, mode=config_mode.mode)

### Main signals


t=capData.t

broadband1=capData.get_signal_by_name('broadband_noise')
broadband2=capData.get_signal_by_name('broadband_noise_bis')
broadband3=capData.get_signal_by_name('broadband_noise_bis_bis')

broadband_avg=(broadband1+broadband2+broadband3)/3

nomasker1=capData.get_signal_by_name('nomasker_bis')

nomasker_avg=(nomasker1)


def plot_main_CAPs(**kwargs):
	pl.figure(**kwargs)
	pl.plot(t*1e3, broadband1*1e3)
	pl.plot(t*1e3, broadband2*1e3)
	pl.plot(t*1e3, broadband3*1e3)
	pl.plot(t*1e3, broadband_avg*1e3, label='avg broadband')


	pl.plot(t*1e3, nomasker1*1e3, label='no masker')

	pl.xlabel('t (ms)')
	pl.ylabel('Amplitude (μV)')
	#pl.xlim([0.004, 0.007])
	pl.legend()
	pl.show()


### Windowing/processing
#NB: 1st processing (windowing + filtering), 2nd processing (diff with broadband condition + smoothing)

t0=5.7e-3
t1=9e-3
ind0=int(t0*48828)

ind0=int(t0*48828)
ind1=int(t1*48828)

win0=sg.tukey(ind1-ind0, alpha=0.4)

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


def process_signal(sig, cumsum=cumsum_default, return_t=False, applyfilter=True):
	
	
	if applyfilter:
		sig=apply_filter(sig)
	sig2=sig*win
	
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
	
def process_signal2(sig, cumsum=cumsum_default, gauss_sigma=0, applyfilter=True):
	'''subtracts the broadband noise response
	gauss_sigma: if diff of 0, smooths the signal with gaussian filter'''
	
	res = process_signal(sig-broadband_avg, cumsum=cumsum, applyfilter=applyfilter)
	if gauss_sigma !=0:
		res = gaussian_filter1d(res, gauss_sigma)
	return res

def plot_CAP_w_wo_filter():
	for plot_time in [False, True]:
		pl.figure()
		for applyfilter in [False, True]:
			t2, broadband_proc=process_signal(broadband_avg, cumsum=cumsum_default, return_t=True, applyfilter=applyfilter)
			nomasker_proc=process_signal(nomasker_avg, applyfilter=applyfilter)
			
			if plot_time:
				pl.plot(t2*1e3, nomasker_proc-broadband_proc, label= 'with filter' if applyfilter else 'without filter')
			else:
				sig2_fft= np.fft.rfft(nomasker_proc-broadband_proc)

				f=np.linspace(0, 48828/2, len(sig2_fft))
				pl.plot(f, np.abs(sig2_fft), label='with filter' if applyfilter else 'without filter')
				pl.xlim([0, 6000])

	pl.legend()
	pl.show()
	
t2, broadband_proc=process_signal(broadband_avg, cumsum=cumsum_default, return_t=True, applyfilter=True)
nomasker_proc=process_signal(nomasker_avg, applyfilter=True)

### Estimation ur / raw excitation pattern

### XXX depends on what is the focus (CF dependent)

#sig=capData.get_signal_by_name('7_notch8000_bw2300_29dB')  #high freq
sig=capData.get_signal_by_name('8_notch4000_bw1700_29dB')
#sig=capData.get_signal_by_name('9_notch3000_bw1500_29dB')
#sig=capData.get_signal_by_name('8_notch2200_bw1500_29dB') #medium freq

sig2=process_signal2(sig, applyfilter=False)

#ur0=sig2-broadband_proc
gauss_sigma=(0.5e-4)/(t2[1]-t2[0])
ur0=process_signal2(sig, gauss_sigma=gauss_sigma, applyfilter=True)
ur0=np.roll(ur0, -100)

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

def proj_E(E, t0=4e-3, t1=8e-3):
	'''
	constraints u between t0 and t1'''
	proj=t2>t0
	proj*=t2<t1
	return E*proj

def deconv_newton(E0, released_sig, ur0=ur0, alpha=0.02, nb_steps=20, eps_ridge=1e-1, verbose=False, t0=4e-3, t1=7e-3):
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
	pl.plot(t2*1e3, E0, label=f'E0 (simple deconv)')
	pl.plot(t2*1e3, E, label=f'E0 (w/ proj)')
	pl.xlabel('t (ms)')
	pl.legend()
	pl.show()


### Narrowband analysis

s1=capData.get_signal_by_name('1_hp_10000Hz')
s2=capData.get_signal_by_name('2_hp_9000Hz')
s3=capData.get_signal_by_name('3_hp_8000Hz')
s4=capData.get_signal_by_name('4_hp_7000Hz')
s5=capData.get_signal_by_name('5_hp_6000Hz')
s6=capData.get_signal_by_name('6_hp_5000Hz')
s7=capData.get_signal_by_name('7_hp_4000Hz')
s8=capData.get_signal_by_name('8_hp_3200Hz')
s9=capData.get_signal_by_name('9_hp_2400Hz')
s10=capData.get_signal_by_name('10_hp_1800Hz')
s11=capData.get_signal_by_name('11_hp_1500Hz')
s12=capData.get_signal_by_name('12_hp_1200Hz')

s1_proc=process_signal(s1)
s2_proc=process_signal(s2)
s3_proc=process_signal(s3)
s4_proc=process_signal(s4)
s5_proc=process_signal(s5)
s6_proc=process_signal(s6)
s7_proc=process_signal(s7)
s8_proc=process_signal(s8)
s9_proc=process_signal(s9)
s10_proc=process_signal(s10)
s11_proc=process_signal(s11)
s12_proc=process_signal(s12)

# pl.figure()
# pl.plot(t2*1e3, s1_proc, label='10 kHz')
# #pl.plot(t2*1e3, s3_proc, label='8 kHz')
# pl.plot(t2*1e3, s5_proc, label='6 kHz')
# #pl.plot(t2*1e3, s7_proc, label='4 kHz')
# pl.plot(t2*1e3, s8_proc, label='3.2 kHz')
# pl.plot(t2*1e3, s9_proc, label='2.4 kHz')
# pl.plot(t2*1e3, s10_proc, label='1.8 kHz')
# pl.plot(t2*1e3, s11_proc, label='1.5 kHz')
# pl.plot(t2*1e3, broadband_proc, label='broadband noise')
# pl.xlabel('t (ms)')
# pl.ylabel('Amplitude')
# pl.xlim([3,12])
# pl.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# pl.show()

s1_proc=process_signal2(s1, gauss_sigma=gauss_sigma)
s2_proc=process_signal2(s2, gauss_sigma=gauss_sigma)
s3_proc=process_signal2(s3, gauss_sigma=gauss_sigma)
s4_proc=process_signal2(s4, gauss_sigma=gauss_sigma)
s5_proc=process_signal2(s5, gauss_sigma=gauss_sigma)
s6_proc=process_signal2(s6, gauss_sigma=gauss_sigma)
s7_proc=process_signal2(s7, gauss_sigma=gauss_sigma)
s8_proc=process_signal2(s8, gauss_sigma=gauss_sigma)
s9_proc=process_signal2(s9, gauss_sigma=gauss_sigma)
s10_proc=process_signal2(s10, gauss_sigma=gauss_sigma)
s11_proc=process_signal2(s11, gauss_sigma=gauss_sigma)
s12_proc=process_signal2(s12, gauss_sigma=gauss_sigma)
	

def plot_figures_narrowband_analysis():
	pl.figure()
	pl.plot(s1_proc)
	pl.plot(s2_proc)
	pl.plot(s3_proc)
	pl.plot(s4_proc)
	pl.plot(s5_proc)
	pl.plot(s6_proc)
	pl.plot(s7_proc)
	pl.plot(s8_proc)
	pl.plot(s9_proc)
	pl.plot(s10_proc)
	pl.plot(s11_proc)
	pl.plot(s12_proc)
	pl.show()

	pl.figure()
	pl.plot(t2*1e3,s1_proc-s2_proc, label='9-10kHz')
	pl.plot(t2*1e3,s2_proc-s3_proc, label='8-9kHz')
	pl.plot(t2*1e3,s3_proc-s4_proc, label='7-8kHz')
	pl.plot(t2*1e3,s4_proc-s5_proc, label='6-7kHz')
	pl.plot(t2*1e3,s5_proc-s6_proc, label='5-6kHz')
	pl.plot(t2*1e3,s6_proc-s7_proc, label='4-5kHz')
	pl.xlim([3,12])
	pl.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
	pl.show()



	pl.figure()
	pl.plot(t2*1e3,s6_proc-s7_proc, label='4-5kHz')
	pl.plot(t2*1e3,s7_proc-s8_proc, label='3.2-4kHz')
	pl.plot(t2*1e3,s8_proc-s9_proc, label='2.4-3.2kHz')
	pl.plot(t2*1e3,s9_proc-s10_proc, label='1.8-2.4kHz')
	pl.plot(t2*1e3,s10_proc-s11_proc, label='1.5-1.8kHz')
	pl.plot(t2*1e3,s11_proc-s12_proc, label='1.2-1.5kHz')
	pl.plot(t2*1e3,s12_proc, label='-1.2kHz')
	pl.xlim([3,12])
	pl.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
	pl.show()


def plot_figures_narrowband_analysis_deconv():
	i=0
	pl.figure(figsize=(10,8))
	for sig, label in [(s1_proc-s2_proc, '9-10kHz'),
	(s2_proc-s3_proc, '8-9kHz'),
	(s3_proc-s4_proc, '7-8kHz'),
	(s4_proc-s5_proc, '6-7kHz'),
	(s5_proc-s6_proc, '5-6kHz'),
	(s6_proc-s7_proc, '4-5kHz'),(s7_proc-s8_proc, '3.2-4kHz'),
	(s8_proc-s9_proc, '2.4-3.2kHz'),
	(s9_proc-s10_proc, '1.8-2.4kHz'),
	(s10_proc-s11_proc, '1.5-1.8kHz'),
	(s11_proc-s12_proc, '1.2-1.5kHz'),
	(s12_proc, '-1.2kHz')]:
	    E=deconv(sig, eps=1e-2)
	    E=deconv_newton(E, sig, alpha=0.005, nb_steps=50, eps_ridge=2e-1, t0=4.3e-3, t1=7e-3)
	    pl.plot(t2*1e3, E-0.25*i, label=label)
	    i+=1

	pl.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
	pl.xlim([4, 8])
	pl.show()



### Latencies

#NB: some data not clear
t_max=np.array([43.5,43.5,45,42,45.5,46.5,48,51,56,65,74,82])
#t_max_bis=np.array([23,26,25.5,25.5,28.5,29])

t_max_C=np.array([44,51.5,47,46,52,54,49,52,65,73,77,96.5])   #mod 0

t_max_R=np.array([44,49.5,47.5,45,49.5,51,53.5,61,76.5,88,104,122])


#t_0lat=4e-3-3.6e-3+2e-3
t_0lat=4e-3+2e-3-6e-3


t_max=t_0lat+t_max*2*1e-5
t_max_C=t_0lat+t_max_C*2*1e-5

t_max_R=t_0lat+t_max_R*2*1e-5

#t_max_bis=t_0lat+t_max_bis*2*1e-5
freqs=np.array([9.5,8.5,7.5,6.5,5.5,4.5,3.6,2.8,2.1,1.65, 1.35, 1])

def plot_estimated_latencies_deconv():
	pl.figure()
	pl.plot(freqs, t_max*1e3, '+', markersize=12, label='C+R')
	pl.plot(freqs, t_max_C*1e3, '+', markersize=12, label='C')

	pl.plot(freqs, t_max_R*1e3, '+', markersize=12, label='R')


	#pl.plot(freqs[0:len(t_max_bis)], t_max_bis*1e3, '+', markersize=12, label='C+R (first peak?)')

	pl.ylabel('Estimated latencies (ms)')
	pl.xlabel('freq (kHz)')

	pl.legend()
	pl.show()
#NB CM begins at 6 ms approx
#peak convol begins at 5-3 ms = 2ms (C) approx


# fit latencies 2 power laws

#above 4kHz


freqs_pts=np.array([9.5,8.5,7.5,6.5,5.5,4.5])*1e3

inds=np.array([0,1,2, 4,5]) #HACK remove outlier 6.5 kHz
freqs_pts=freqs_pts[inds]
freqs_pts0=freqs_pts
t_max_pts=t_max[inds]
t_max_pts0=t_max_pts


lat=PowerLawLatencies(1e6, alpha=1, t0=4e-3, mode='left')
lat.fit_data(t_max_pts, freqs_pts, init_with_new_values=False, bounds=[0.5, 2])

lat_above4k=lat


#below 4kHz
freqs_pts=freqs[6:]*1e3
t_max_pts=t_max[6:]
lat=PowerLawLatencies()
lat.fit_data(t_max_pts, freqs_pts)

def plot_latencies_fit():

	pl.figure()
	freqs_lin=np.linspace(4, 10)*1e3
	pl.plot(freqs_lin, lat(freqs_lin)*1e3)

	pl.plot(freqs_pts, t_max_pts*1e3, '+', markeredgewidth=3, markersize=10)

	freqs_lin=np.linspace(0.5, 5)*1e3
	pl.plot(freqs_lin, lat(freqs_lin)*1e3, color='C2')


	freqs_lin=np.linspace(0.5, 10)*1e3
	pl.plot(freqs_lin, lat(freqs_lin)*1e3, color='C2', linestyle='--')

	pl.plot(freqs_pts, t_max_pts*1e3, '+', markeredgewidth=3, markersize=10)

	pl.show()

	pl.figure()

	pl.plot(freqs_lin, (lat(freqs_lin)-lat.t0)*1e3, color='C2', linestyle='--')

	pl.plot(freqs_pts0, (t_max_pts0-lat.t0.numpy())*1e3, '+', markeredgewidth=3, markersize=10)

	pl.plot(freqs_pts, (t_max_pts-lat.t0.numpy())*1e3, '+', markeredgewidth=3, markersize=10)

	pl.ylabel(' t - t_0 (ms)')

	pl.xlabel(' f (Hz)')

	pl.xscale('log')
	pl.yscale('log')
	pl.show()

