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

	pl.figure()
	pl.plot(f,g)
	pl.xlabel('f (Hz)')
	pl.ylabel('Amplitude spectrum')
	pl.show()

	g_dB=auditoryFilter.g_dB(f)
	pl.figure()
	pl.plot(f,g_dB)
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
	#rectFilter = AuditoryFilter(300)
	#plot_auditory_filter(rectFilter)

	#gaussianFilter= GaussianFilter(115.)
	#plot_auditory_filter(gaussianFilter)
	
	# gaussianFilter2= GaussianFilter.givenQ10(1e3, 2.)
	# plot_auditory_filter(gaussianFilter2)
	

	#test_g_db(gaussianFilter)



	#test_L2_normalization(rectFilter)
	#test_L2_normalization(gaussianFilter)


########## MASK DEGREE FUNC ##############

def plotMaskingDegreeFunc(maskingDegreeFunction):
	I=np.linspace(0, 100, num=500)

	pl.figure()
	pl.plot(I, maskingDegreeFunction.md(I)*100)
	pl.xlabel('I (dB)')
	pl.ylabel('masking degree (%)')
	pl.ylim([0, 100])
	pl.show()


if __name__ == '__main__':
	pass
	lin=MaskingDegreeFunction(20, 40, 0.8)
	plotMaskingDegreeFunc(lin)

	sig=SigmoidMaskingDegreeFunction(30, 2*1/15., 0.8)
	plotMaskingDegreeFunc(sig)



########## UNITARY RESP. ##############

def plotUR(t, u, name=''):
	pl.figure()
	pl.title(f'Unitary response - {name}')

	pl.plot(t*1e3,u)
	pl.xlabel('t (ms)')
	pl.ylabel('Amplitude')

	pl.show()


if __name__ == '__main__':
	pass
	ur=URWang1979
	t = np.linspace(ur._t[0], ur._t[-1], num=500) #resample t
	u = ur.u(t)
	plotUR(t,u, name=ur.name)




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

	lat3=Eggermont1976clickLatencies80dB
	print(lat3)
	plotLatencies(lat3)


####### EXCITATION PATTERNS ########

def plotExcitationPattern(EPat):

	masker=EPat.masker
	lat = EPat.latencies

	t=np.linspace(5e-4, 10e-3, num=500)
	f=lat.f_from_t(t)

	pl.figure()
	pl.subplot(1,2,1)
	pl.plot(t*1000,EPat.E0(t))
	pl.twinx()
	pl.plot(t*1000,1-masker.M(f), color='r')
	pl.ylim([0,1.1])
	pl.xlabel('t (ms)')
	pl.title('Raw excitation pattern and masking pattern (1-M)')

	locs =np.linspace(1, 10, num=10)
	pl.gca().set_xticks(locs)   

	ax2 = pl.gca().twiny()
	ax2.set_xticks(locs)
	ax2.set_xticklabels(np.round(lat.f_from_t(locs*1e-3)/50).astype(np.int32)*50)
	ax2.set_xlabel('f (Hz)')


	pl.subplot(1,2,2)

	pl.plot(t*1000,EPat.E(t))
	pl.plot(t*1000,EPat.E0(t)*(1-masker.M(f)), color='r')
	pl.xlabel('t (ms)')
	pl.title('Masked masking pattern E*(1-M)')

	pl.gca().set_xticks(locs)   

	ax2 = pl.gca().twiny()
	ax2.set_xticks(locs)
	ax2.set_xticklabels(np.round(lat.f_from_t(locs*1e-3)/50).astype(np.int32)*50)
	ax2.set_xlabel('f (Hz)')

	pl.show()





if __name__ == '__main__':
	lat=Eggermont1976clickLatencies80dB
	EPat = ExcitationPattern(lat, 3000, 1.5, 70)

	plotExcitationPattern(EPat)