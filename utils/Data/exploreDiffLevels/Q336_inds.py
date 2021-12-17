#dic atten -> list tuples (begin_ind, end_ind)
dic_inds={
 40: [(1487, 1534), (1782, 1829), (2070, 2117), (2358, 2405), (2646, 2693)] ,
 35: [(1535, 1582), (1830, 1877), (2118, 2165), (2406, 2453), (2694, 2741)],
 30: [(1583, 1630), (1878, 1925), (2166, 2213), (2454, 2501), (2742, 2789)],
 25: [(1631, 1678), (1926, 1973), (2214, 2261), (2502, 2549), (2790, 2837)],
 20: [(1686, 1733), (1974, 2021), (2262, 2309), (2550, 2597), (2838, 2885)],
 15: [(1734, 1781), (2022, 2069), (2310, 2357), (2598, 2645), (2886, 2933)]
 }

#1679-1685: wront stim


#list maskers + ref maskers


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