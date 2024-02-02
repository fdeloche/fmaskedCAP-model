

import numpy as np

import scipy.signal as sg

from functools import partial



#first filter

fs = 48828  # Sample frequency (Hz)
f0 = 1500  # Frequency to be removed from signal (Hz)
Q = 7 # Quality factor
# Design notch filter
b, a = sg.iirnotch(f0, Q, fs)


# # Frequency response
# freq, h = sg.freqz(b, a, fs=fs)
# # Plot
# fig, ax = pl.subplots(2, 1, figsize=(8, 6))
# ax[0].plot(freq, 20*np.log10(abs(h)), color='blue')
# ax[0].set_title("Frequency Response")
# ax[0].set_ylabel("Amplitude (dB)", color='blue')
# ax[0].set_xlim([0, 4000])
# ax[0].set_ylim([-25, 10])
# ax[0].grid()
# ax[1].plot(freq, np.unwrap(np.angle(h))*180/np.pi, color='green')
# ax[1].set_ylabel("Angle (degrees)", color='green')
# ax[1].set_xlabel("Frequency (Hz)")
# ax[1].set_xlim([0, 4000])
# ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
# ax[1].set_ylim([-90, 90])
# ax[1].grid()
# pl.show()


#second filter

fs = 48828  # Sample frequency (Hz)
f0 = 4500  # Frequency to be removed from signal (Hz)
Q = 5  # Quality factor
# Design notch filter
b2, a2 = sg.iirnotch(f0, Q, fs)
# # Frequency response
# freq, h = sg.freqz(b2, a2, fs=fs)
# # Plot
# fig, ax = pl.subplots(2, 1, figsize=(8, 6))
# ax[0].plot(freq, 20*np.log10(abs(h)), color='blue')
# ax[0].set_title("Frequency Response")
# ax[0].set_ylabel("Amplitude (dB)", color='blue')
# ax[0].set_xlim([0, 8000])
# ax[0].set_ylim([-25, 10])
# ax[0].grid()
# ax[1].plot(freq, np.unwrap(np.angle(h))*180/np.pi, color='green')
# ax[1].set_ylabel("Angle (degrees)", color='green')
# ax[1].set_xlabel("Frequency (Hz)")
# ax[1].set_xlim([0, 8000])
# ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
# ax[1].set_ylim([-90, 90])
# ax[1].grid()
# pl.show()


def apply_filter_aux(b, a, b2, a2, sig):
    
    sig2=sg.lfilter(b, a, sig)
    
    #sig2=sg.lfilter(b, a, sig2) #apply second time
    sig2=sg.lfilter(b2, a2, sig2)
    return sig2

apply_filter=partial(apply_filter_aux, b, a, b2, a2)