'''For testing purpose.'''

from filters import *

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

if __name__ == '__main__':
	#rectFilter = AuditoryFilter(300)
	#plot_auditory_filter(rectFilter)

	gaussianFilter= GaussianFilter(125.)
	plot_auditory_filter(gaussianFilter)

	#TODO test normalization filters