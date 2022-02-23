#dic atten -> list tuples (begin_ind, end_ind)
dic_inds={
 45: [(583, 630), (871, 918), (1159, 1206), (1447, 1494), (1735, 1782), (2023, 2070)] ,
 40: [(631, 678), (919, 966), (1207, 1254), (1495, 1542), (1783, 1830), (2071, 2118)],
 35: [(679, 726), (967, 1014), (1255, 1302), (1543, 1590), (1831, 1878), (2119, 2166)],
 30: [(727, 774), (1015, 1062), (1303, 1350), (1591, 1638), (1879, 1926), (2167, 2214)],
 25: [(775, 822), (1063, 1110), (1351, 1398), (1639, 1686), (1927, 1974), (2215, 2262)],
 20: [(823, 870), (1111, 1158), (1399, 1446), (1687, 1734), (1975, 2022), (2263, 2310)]
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


#more explicit labels for hp6200 gradual amp
labels_gradual_amp={
 '11_hp6200_gradualamp':  '11_hp6200_gradualamp_bw1000',
 '12_hp6200_gradualamp':  '12_hp6200_gradualamp_bw700',
 '13_hp6200_gradualamp':  '13_hp6200_gradualamp_bw1400',
 '14_hp6200_gradualamp':  '14_hp6200_gradualamp_bw1200',
 '15_hp6200_gradualamp':  '15_hp6200_gradualamp_nonotch',
 '16_hp6200_gradualamp':  '16_hp6200_gradualamp_nonotch_27dB',
}
