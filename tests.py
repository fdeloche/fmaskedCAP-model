'''For testing purpose.'''

from filters import *
from masking import *
from simulCAP import *

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
	rectFilter = AuditoryFilter(300)
	#plot_auditory_filter(rectFilter)

	gaussianFilter= GaussianFilter(115.)
	plot_auditory_filter(gaussianFilter)
	
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
	#lin=MaskingDegreeFunction(20, 40, 0.8)
	#plotMaskingDegreeFunc(lin)

	#sig=SigmoidMaskingDegreeFunction(30, 2*1/15., 0.8)
	#plotMaskingDegreeFunc(sig)



########## UNITARY RESP. ##############

def plotUR(t, u):
	pl.figure()

	pl.plot(t*1e3,u)
	pl.xlabel('t (ms)')
	pl.ylabel('Amplitude')

	pl.show()


if __name__ == '__main__':
	pass
	# ur=URfromCsv('./UR/Wang1979Fig14.csv')
	# t = np.linspace(ur._t[0], ur._t[-1], num=500) #resample t
	# u = ur.u(t)
	# plotUR(t,u)

