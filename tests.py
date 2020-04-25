'''For testing purpose.'''

from filters import *
from masking import *
from simulCAP import *
from latencies import *
from excitation import *

import numpy as np
import matplotlib.pyplot as pl

################ AUDITORY FILTERS ######################

def plot_auditory_filter(auditoryFilter):
	print(auditoryFilter)

	f=np.linspace(-2000, 2000, num=500)
	g=auditoryFilter.g(f)

	right_int=auditoryFilter.right_sq_int(f)
	pl.figure()
	pl.plot(f,g, label='filter shape')
	pl.xlabel('f (Hz)')
	pl.ylabel('Amplitude spectrum')
	pl.legend()
	pl.gca().twinx()
	pl.plot(f,right_int, label='right integral', color='orange')
	# #Check with error function
	# from scipy.special import erf
	# sig=auditoryFilter.BW10()/(2*2.14)
	# pl.plot(f, 0.5-0.5*erf(f/(np.sqrt(2)*sig)))
	pl.legend(loc='lower right')
	pl.show()

	g_dB=auditoryFilter.g_dB(f)
	right_int_dB=auditoryFilter.right_sq_int_dB(f)
	pl.figure()
	pl.plot(f,g_dB, label='filter shape')

	pl.plot(f,right_int_dB, label='right integral', color='orange')
	pl.xlabel('f (Hz)')
	pl.ylabel('Amplitude spectrum (dB)')
	pl.ylim([-100,20])
	pl.show()

def test_L2_normalization(auditoryFilter, eps=1e-2):
	f=np.linspace(-2000, 2000, num=5000)
	g=auditoryFilter.g(f)

	g_l2=np.trapz(g**2, dx=f[1]-f[0])

	assert np.abs(g_l2-1.)<eps, f"g not normalized (g_l2: {g_l2:.4f})"

def test_g_db(auditoryFilter):
	print(auditoryFilter)

	f=np.linspace(-2000, 2000, num=500)
	g=auditoryFilter.g(f)

	g_dB=auditoryFilter.g_dB(f)
	pl.figure()
	pl.plot(f,g, label='g')
	pl.plot(f, np.power(10, g_dB/20), label='g from g_dB')
	pl.xlabel('f (Hz)')
	pl.ylabel('Amplitude spectrum (dB)')
	pl.gca().legend()
	pl.show()



if __name__ == '__main__':
	pass
	# #rectFilter = AuditoryFilter(300)
	# #plot_auditory_filter(rectFilter)

	# #gaussianFilter= GaussianFilter(115.)
	# #plot_auditory_filter(gaussianFilter)
	
	# gaussianFilter2= GaussianFilter.givenQ10(1e3, 2.)
	# plot_auditory_filter(gaussianFilter2)
	

	# #test_g_db(gaussianFilter)



	# #test_L2_normalization(rectFilter)
	# #test_L2_normalization(gaussianFilter)


########## MASK DEGREE FUNC ##############

def plotMaskingDegreeFunc(maskingDegreeFunction):
	I=np.linspace(0, 100, num=500)

	pl.figure()
	pl.plot(I, maskingDegreeFunction.md(I)*100)
	pl.xlabel('Spectral intensity (dB)')
	pl.ylabel('masking degree (%)')
	pl.ylim([0, 100])
	pl.show()


if __name__ == '__main__':
	pass
	# lin=MaskingDegreeFunction(20, 40, 0.8)
	# plotMaskingDegreeFunc(lin)

	# sig=SigmoidMaskingDegreeFunction(30, 2*1/15., 0.8)
	# plotMaskingDegreeFunc(sig)



########## UNITARY RESP. ##############

def plotUR(t, u, name=''):
	pl.figure()
	pl.title(f'Unitary response - {name}')

	pl.subplot(1, 2, 1)
	pl.plot(t*1e3,u)
	pl.xlabel('t (ms)')
	pl.ylabel('Amplitude')


	pl.subplot(1, 2, 2)
	dt=t[1]-t[0]
	fny= 1./(2*dt)
	f=np.linspace(0, fny, num=len(t)//2+1)
	u_fft=np.sqrt(t[-1]+dt)/len(t)*np.fft.rfft(u)
	pl.plot(f,np.abs(u_fft))
	pl.xlabel('f (Hz)')
	pl.ylabel('Amplitude')
	pl.xlim([0,3000])
	pl.show()


def plotURs(t, urlist):
	pl.figure()
	pl.title(f'Unitary responses')
	pl.subplot(1,2,1)
	for ur in urlist:
		pl.plot(t*1e3,ur.u(t), label=ur.name)
	pl.xlabel('t (ms)')
	pl.ylabel('Amplitude')
	pl.legend()



	pl.subplot(1, 2, 2)
	dt=t[1]-t[0]
	fny= 1./(2*dt)
	f=np.linspace(0, fny, num=len(t)//2+1)
	for ur in urlist:
		u=ur.u(t)
		u_fft=np.sqrt(t[-1]+dt)/len(t)*np.fft.rfft(u)
		pl.plot(f,np.abs(u_fft), label=ur.name)
	pl.xlabel('f (Hz)')
	pl.ylabel('Amplitude')
	pl.xlim([0,3000])
	pl.show()


if __name__ == '__main__':
	pass
	# ur=URWang1979
	# ur2=URWang1979m
	# t = np.linspace(ur._t[0], ur._t[-1], num=512) #resample t
	# # u = ur.u(t)
	# # plotUR(t,u, name=ur.name)
	# plotURs(t,[ur, ur2])




########## LATENCIES ##############


def plotLatencies(lat):
	t0=lat.t0

	tmin=lat.t_from_f(10e3)

	t=np.linspace(tmin-t0, 10e-3-t0) #t - t0
	f=lat.f_from_t(t+t0)

	pl.figure()
	pl.title(f'Latencies {lat.name}')
	pl.plot(t, f)
	pl.xlabel('t-t0 (s)')
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


if __name__ == '__main__':
	pass
	# lat=PowerLawLatencies.fromPts(2e-3, 8000, 5e-3, 1000)
	# lat2=PowerLawLatencies.shift(lat, 1e-3)
	# print(lat2)
	# plotLatencies(lat2)

	# #test f from t, t from f
	# f=2678
	# f2 = lat2.f_from_t(lat2.t_from_f(f))
	# assert np.abs(f-f2)<1e-1

	# lat3=Eggermont1976clickLatencies80dB
	# print(lat3)
	# plotLatencies(lat3)


####### EXCITATION PATTERNS ########

def plotExcitationPattern(EPat, **kwargsplot):

	masker=EPat.masker
	lat = EPat.latencies

	t=np.linspace(5e-4, 10e-3, num=500)
	f=lat.f_from_t(t)

	pl.figure(**kwargsplot)
	pl.subplot(1,2,1)
	pl.plot(t*1000,EPat.E0(t))
	pl.twinx()
	pl.plot(t*1000,1-masker.M(f), color='r')
	pl.ylim([0,1.1])
	if isinstance(masker, ToneSingleFilterMaskingPattern):
		pl.text(5, 0.8, f'f_m={masker.f_0/1e3:.2f} kHz\nQ10={masker.f_0/masker.filt.BW10():.1f}')
	if isinstance(masker, HighPassNoiseSingleFilterMaskingPattern):
		pl.text(5, 0.8, f'f_cut={masker.f_cut/1e3:.2f} kHz\nQ10={EPat.f_c/masker.filt.BW10():.1f} (at max excitation)')
	pl.xlabel('t (ms)')
	pl.title('Raw excitation pattern\n& masking pattern (1-M)')

	locs =np.linspace(1, 10, num=10)
	pl.gca().set_xticks(locs)   

	ax2 = pl.gca().twiny()
	ax2.set_xticks(locs)
	ax2.set_xticklabels([f'{CF/1e3:.1f}' for CF in list(lat.f_from_t(locs*1e-3))])
	ax2.set_xlabel('Place: CF (kHz)')


	pl.subplot(1,2,2)

	pl.plot(t*1000,EPat.E(t))
	#pl.plot(t*1000,EPat.E0(t)*(1-masker.M(f)), color='r')
	pl.xlabel('t (ms)')
	pl.title('Masked excitation pattern E*(1-M)')

	pl.gca().set_xticks(locs)   

	ax2 = pl.gca().twiny()
	ax2.set_xticks(locs)
	ax2.set_xticklabels([f'{CF/1e3:.1f}' for CF in list(lat.f_from_t(locs*1e-3))])
	ax2.set_xlabel('Place: CF (kHz)')


	pl.show()





if __name__ == '__main__':
	pass
	# lat=Eggermont1976clickLatencies80dB
	# EPat = ExcitationPattern(lat, 3000, 1.5, 70)

	# # print(EPat)
	# # plotExcitationPattern(EPat)

	# gaussianFilter= GaussianFilter.givenQ10(3e3, 3.)

	# #md=linMD=MaskingDegreeFunction(20, 40, 0.8)
	# md=sig=SigmoidMaskingDegreeFunction(30, 2*1/15., 0.8)
	# plotMaskingDegreeFunc(md)
	# A=10**(35/20.)*np.sqrt(gaussianFilter.ERB())


	# #mask= ToneSingleFilterMaskingPattern(A, 3e3, gaussianFilter, md)
	# mask= HighPassNoiseSingleFilterMaskingPattern(40, 3e3, gaussianFilter, md)
	
	# EPat2 = ExcitationPattern.mask(EPat, mask)

	# print(EPat2)
	# plotExcitationPattern(EPat2)



#### SIMUL CAP: double convolution (w/ Gaussian Kernel) ###


def plotSimulCAP2convGaussianKernel(EPat, sig=5e-4):

	masker=EPat.masker
	lat = EPat.latencies

	t=np.linspace(5e-4, 10e-3, num=500)
	f=lat.f_from_t(t)

	pl.figure()

	pl.subplot(1,2,1)
	Em=EPat.E(t)
	pl.plot(t*1000,Em)
	#pl.plot(t*1000,EPat.E0(t)*(1-masker.M(f)), color='r')
	pl.xlabel('t (ms)')
	pl.title('Masked excitation patterns E*(1-M)')

	locs =np.linspace(1, 10, num=10)
	pl.gca().set_xticks(locs)   

	if isinstance(masker, ToneSingleFilterMaskingPattern):
		pl.text(5, 0.8*np.amax(Em), f'f_m={masker.f_0/1e3:.2f} kHz\nQ10={masker.f_0/masker.filt.BW10():.1f}')

	ax2 = pl.gca().twiny()
	ax2.set_xticks(locs)
	ax2.set_xticklabels(np.round(lat.f_from_t(locs*1e-3)/50).astype(np.int32)*50)
	ax2.set_xlabel('f (Hz)')


	pl.subplot(1,2,2)
	pl.title('Simulated CAP')

	cap0 = simulCAP2convolGaussianKernel(ExcitationPattern.rawExcitationPattern(EPat), t, sig=sig)
	cap = simulCAP2convolGaussianKernel(EPat, t, sig=sig)

	pl.plot(t*1000,cap0, label='raw')
	pl.plot(t*1000,cap, label='masked')
	pl.xlabel('t (ms)')

	pl.legend()

	locs =np.linspace(1, 10, num=10)
	pl.gca().set_xticks(locs)   

	ax2 = pl.gca().twiny()
	ax2.set_xticks(locs)
	ax2.set_xticklabels(np.round(lat.f_from_t(locs*1e-3)/50).astype(np.int32)*50)
	ax2.set_xlabel('f (Hz)')


	pl.show()




def plotSimulCAPs2convGaussianKernel(EPats, sig=5e-4, secondPeak=0., plotGaussianKernel=True, plotmaskingpattern=False, **kwargsplots):
	lat = EPats[0].latencies

	t_max=10e-3
	t_min=5e-4

	t=np.linspace(t_min,t_max, num=500)
	f=lat.f_from_t(t)

	pl.figure(**kwargsplots)


	#EXCITATION + MASKING PATTERNS : temporal

	pl.subplot(2,2,1)
	
	pl.plot(t*1000,EPats[0].E0(t), label=f'raw')
	for EPat in EPats:
		masker=EPat.masker

		Em=EPat.E(t)
		Q10=EPat.f_c/masker.filt.BW10()
		pl.plot(t*1000,Em, label=f'Q10: {Q10:.2f}')
		if plotmaskingpattern:
			pl.gca().twinx()
			pl.plot(t*1000, 1-masker.M(f), linestyle='--', color='black', label='masking pattern')
			pl.legend()

	pl.xlabel('t (ms)')
	pl.title('Masked excitation patterns E*(1-M)')

	locs =np.linspace(1, 10, num=10)
	pl.gca().set_xticks(locs)   

	#pl.legend()
	ax2 = pl.gca().twiny()
	ax2.set_xticks(locs)
	ax2.set_xticklabels([f'{CF/1e3:.1f}' for CF in list(lat.f_from_t(locs*1e-3))])
	ax2.set_xlabel('Place: CF (kHz)')


	#EXCITATION + MASKING PATTERNS : in frequency


	pl.subplot(2,2,3)
	
	dt=t[1]-t[0]
	fny=1./(2*dt)
	f=np.linspace(0, fny, len(t)//2+1)
	norm_c = np.sqrt(t_max-t_min)/len(t)
	E0=EPats[0].E0(t)
	E0_fft=norm_c*np.fft.rfft(E0)
	pl.plot(f, np.abs(E0_fft), label=f'raw')
	for EPat in EPats:
		masker=EPat.masker
		Em=EPat.E(t)
		Q10=EPat.f_c/masker.filt.BW10()
		Em_fft=norm_c*np.fft.rfft(Em)
		pl.plot(f,np.abs(Em_fft), label=f'Q10: {Q10:.2f}')

	pl.xlabel('f (Hz)')
	#pl.legend()

	pl.xlim([0,2000])

	#SIMUL CAP : temporal

	pl.subplot(2,2,2)
	pl.title('Simulated CAPs')

	cap0 = simulCAP2convolGaussianKernel(ExcitationPattern.rawExcitationPattern(EPats[0]), t, sig=sig, secondPeak=secondPeak)
	pl.plot(t*1000,cap0, label='raw')



	for EPat in EPats:
		cap = simulCAP2convolGaussianKernel(EPat, t, sig=sig, secondPeak=secondPeak)
		masker=EPat.masker
		Q10=EPat.f_c/masker.filt.BW10()
		pl.plot(t*1000,cap, label=f'Q10: {Q10:.2f}')

	#also plot Gaussian kernel
	if plotGaussianKernel:
		a=np.amax(cap0)
		g_kn = 0.4*a*np.exp(- (t-(1e-3+3*sig))**2/(2*sig**2))
		pl.plot(t*1000, g_kn, label='Gaussian kernel', color='black', ls='--')

	pl.xlabel('t (ms)')

	#pl.legend()

	locs =np.linspace(1, 10, num=10)
	pl.gca().set_xticks(locs)   

	ax2 = pl.gca().twiny()
	ax2.set_xticks(locs)
	ax2.set_xticklabels([f'{CF/1e3:.1f}' for CF in list(lat.f_from_t(locs*1e-3))])
	ax2.set_xlabel('Place: CF (kHz)')


	# SIMUL CAP : in frequency

	cap0_fft=norm_c*np.fft.rfft(cap0)
	pl.subplot(2,2,4)
	pl.plot(f, np.abs(cap0_fft), label=f'raw')
	for EPat in EPats:
		masker=EPat.masker

		cap = simulCAP2convolGaussianKernel(EPat, t, sig=sig,  secondPeak=secondPeak)
		Q10=EPat.f_c/masker.filt.BW10()
		cap_fft=norm_c*np.fft.rfft(cap)
		pl.plot(f,np.abs(cap_fft), label=f'Q10: {Q10:.2f}')


	#also plot Gaussian kernel
	if plotGaussianKernel:
		g_kn_fft=norm_c*np.fft.rfft(g_kn)
		pl.plot(f,np.abs(g_kn_fft), label='Gaussian kernel', color='black', ls='--')


	pl.xlabel('f (Hz)')
	pl.legend()

	pl.xlim([0,2000])
	pl.show()

if __name__ == '__main__':
	pass
	# lat=Eggermont1976clickLatencies80dB


	# EPat = ExcitationPattern(lat, 6000, 3., 70)


	# # gaussianFilter= GaussianFilter.givenQ10(3e3, 3.)

	# # #md=linMD=MaskingDegreeFunction(20, 40, 0.8)
	# # md=sig=SigmoidMaskingDegreeFunction(30, 2*1/15., 0.8)
	# # #plotMaskingDegreeFunc(md)
	# # A=10**(35/20.)*np.sqrt(gaussianFilter.ERB())

	# # mask= ToneSingleFilterMaskingPattern(A, 3e3, gaussianFilter, md)
	# # EPat2 = ExcitationPattern.mask(EPat, mask)


	# # plotSimulCAP2convGaussianKernel(EPat, sig=2e-4)
	# # plotSimulCAP2convGaussianKernel(EPat2, sig=2e-4)


	# f_m=6e3
	# gaussianFilters= [GaussianFilter.givenQ10(f_m, 4*1.2**i) for i in range(5)]

	# md=sig=SigmoidMaskingDegreeFunction(30, 2*1/15., 0.95)
	# plotMaskingDegreeFunc(md)

	# A=10**(45/20.)*np.sqrt(gaussianFilters[-1].ERB())

	# maskers=[ToneSingleFilterMaskingPattern(A, f_m, gf, md) for gf in gaussianFilters]
	# #maskers=[HighPassNoiseSingleFilterMaskingPattern(50, f_m, gf, md) for gf in gaussianFilters]
	

	# EPats = [ExcitationPattern.mask(EPat, mask) for mask in maskers]

	# plotSimulCAPs2convGaussianKernel(EPats, sig=3e-4, secondPeak=0.)

	# # A2=10**(35/20.)*np.sqrt(gaussianFilters[-1].ERB())

	# # maskers2=[ToneSingleFilterMaskingPattern(A2, f_m, gf, md) for gf in gaussianFilters]
	# # EPats2 = [ExcitationPattern.mask(EPat, mask) for mask in maskers2]

	# #plotSimulCAPs2convGaussianKernel(EPats+EPats2, sig=6e-4)





#### Test class ConvolutionCAPSimulatorSingleFilterModel ####




def plotConvCAPSimulator(capSimul, **kwargsplots):
	
	m=capSimul.m
	t=capSimul.t
	MPs=capSimul.maskingPatterns
	EPs=capSimul.getEPs()
	CAPs=capSimul.simulCAPs()
	lat=capSimul.latencies
	MCs=capSimul.maskingConditions_init

	#TODO
	#also plot ur

	ERB=capSimul.filt.ERB()
	
	pl.figure(**kwargsplots)
	pl.suptitle(f'Simulation of CAPs ERB={ERB:.0f} Hz')
	nb_col=(m+1)//2 if m<=12 else (m+2)//3
	nb_row=2*(m+nb_col-1)//nb_col
	EPamax=np.amax(EPs)
	for i in range(m):

		ind=i+1+(i//nb_col)*nb_col

		pl.subplot(nb_row, nb_col, ind)
		pl.plot(t*1e3, EPs[i])
		pl.xlabel('t (ms)')
		pl.ylim([0, 1.1*EPamax])
		pl.gca().get_yaxis().set_visible(False)
		if isinstance(MCs[i], NoMaskingCondition):
			pl.text(4, 0.5*EPamax, f'raw\n(no masker)')		
		if isinstance(MCs[i], ToneSingleFilterMaskingCondition):
			pl.text(4, 0.5*EPamax, f'tone\n{MCs[i].f_0/1e3:.2f}kHz\nA={20*np.log10(MCs[i].A):.0f}dB')
		if isinstance(MCs[i], HighPassNoiseSingleFilterMaskingCondition):
			pl.text(4, 0.5*EPamax, f'hp noise\n{MCs[i].f_cut/1e3:.2f}kHz\n{MCs[i].IHz:.0f}dB/Hz')
		pl.subplot(nb_row, nb_col, ind+nb_col)
		pl.plot(t*1e3, CAPs[i], color='r')
		pl.gca().get_yaxis().set_visible(False)
		if i==0:
			u=capSimul.u
			pl.plot(t*1e3, capSimul.u*0.4/np.amax(u)*np.amax(CAPs[i]), color='black') #also plot u
		pl.xlabel('t (ms)')
	pl.show()


if __name__ == '__main__':
	pass
	lat=Eggermont1976clickLatencies80dB

	t=ConvolutionCAPSimulatorSingleFilterModel.default_t_array()
	ur=URWang1979m
	u0=ur.u(t-2e-3) #shift
	from scipy.ndimage  import gaussian_filter1d
	sig=0.3e-3 #std of gaussian kernel in s
	sig2=sig/(t[1]-t[0])
	u = gaussian_filter1d(u0, sigma=sig2)

	EPinit = ExcitationPattern(lat, 3000, 2., 70)
	f_m = 3000
	gf= GaussianFilter.givenQ10(f_m, 4.)
	md=sig=SigmoidMaskingDegreeFunction(30, 2*1/15., 0.95)
	#plotMaskingDegreeFunc(md)
	
	A=10**(45/20.)*np.sqrt(gf.ERB())
	mc_1 = ToneSingleFilterMaskingCondition(A, f_m)
	mc_2 = HighPassNoiseSingleFilterMaskingCondition(50, f_m)

	mcs = [NoMaskingCondition(), mc_1, mc_2]


	capSimulator = ConvolutionCAPSimulatorSingleFilterModel(lat, gf, EPinit, md, mcs, ur=u)

	plotConvCAPSimulator(capSimulator)

