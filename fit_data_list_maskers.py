#NOTE:masker names can be in the right format or with the json extension at the end (filenames)
#ex how to handle this:
#masker_list=[st.replace('-', '_').replace('.json', '') for st in fln_list]

import numpy as np

#8k

ntch_8k_masker_list=[
 '5_notch8000_bw2300_40dB',
 '6_notch8000_bw2300_35dB',
 '7_notch8000_bw2300_32dB',
 '7_notch8000_bw2300_29dB',
 '8_notch8000_bw2300_26dB',
 '9_notch8000_bw2300_23dB',
 '10_notch8000_bw2300_17dB',  
 '2-notch7300_1600_30dB.json', 
 '3-notch7300_1600_25dB.json', 
 '4-notch8300_2000_26dB.json',
 '5-notch8300_2000_23dB.json'
 ] 


attns_8k=np.array([40,35,32,29,26,23, 17])

ntch_8k_re='.*notch8000_bw2300'   #not used anymore


vfreq_8k_fln_list=[
'6_notch8000_bw2300_35dB', '8_notch8000_bw2300_26dB',
'1-notch7600_bw1100.json', '2-notch7800_bw1300.json',
'3-notch8000_bw1400.json',
'4-notch8200_bw1300.json',
'5-notch8200_bw1500.json',
'6-notch7900_bw1600.json',
'7-notch8100_bw1200.json',         
'6_notch6000_bw2000_35dB', '9_notch6000_bw2000_26dB',
'2-notch6200_bw1000.json', '3-notch6000_bw1100.json', '9-notch6300_bw1100.json'
'6_notch5000_bw2000_35dB',
'9_notch5000_bw2000_26dB',
'4-notch4800_bw1400.json'

 '2-notch7300_1600_30dB.json', 
 '4-notch8300_2000_26dB.json',
 '1-notch7400_bw1000.json',
'2-notch8400_bw1200.json',
'3-notch7800_bw1800.json',
'8-notch8100_bw2000.json'


'1-notch5800_1400_30dB.json', 
'3-notch6500_1400_26dB.json',
'1-notch6600_bw900.json',
'2-notch7000_bw1200.json'
]

vbw_8k_fln_list=['1-notch7600_bw1100.json',
'2-notch7800_bw1300.json',
'3-notch8000_bw1400.json',
'4-notch8200_bw1300.json',
'5-notch8200_bw1500.json',
'6-notch7900_bw1600.json',
'7-notch8100_bw1200.json']


vbw_8k_fln_list+=['1-notch7400_bw1000.json',
'2-notch8400_bw1200.json',
'3-notch7800_bw1800.json',
'8-notch8100_bw2000.json'] 

vbw_8k_fln_list+= ['2-notch7300_1600_30dB.json', 
 '4-notch8300_2000_26dB.json']


# 6k

ntch_6k_masker_list=[
'4_notch6000_bw2000_45dB',
'5_notch6000_bw2000_40dB',
'6_notch6000_bw2000_35dB',
'7_notch6000_bw2000_32dB',
'8_notch6000_bw2000_29dB',
'9_notch6000_bw2000_26dB',
'10_notch6000_bw2000_23dB',
'11_notch6000_bw2000_17dB', #re-added
'1-notch5800_1400_30dB.json', #new ones
'2-notch5800_1400_25dB.json',
'3-notch6500_1400_26dB.json',
'4-notch6500_1400_23dB.json',


'5-notch7000_1600_26dB.json']



attns_6k=np.array([45,40,35,32,29,26,23,17])

ntch_6k_re='.*notch6000_bw2000'  #not used anymore

vfreq_6k_fln_list=[
           '6_notch8000_bw2300_35dB', '8_notch8000_bw2300_26dB',

'6_notch6000_bw2000_35dB', 
'9_notch6000_bw2000_26dB',
'6_notch5000_bw2000_35dB',
'9_notch5000_bw2000_26dB',
'6_notch4000_bw1700_35dB',
'9_notch4000_bw1700_26dB',

'1-notch7600_bw1100', '2-notch7800_bw1300',
'3-notch8000_bw1400',
'5-notch8200_bw1500',
'1-notch5900_bw900',
'2-notch6200_bw1000',
'3-notch6000_bw1100',
'4-notch6100_bw1200',
'7-notch5900_bw1300',
'8-notch5800_bw1400',
'9-notch6300_bw1100',


'1-notch4800_bw900',
'3-notch5100_bw1200',
'6-notch4900_bw1300',
'7-notch5200_bw900',
'4-notch4200_bw1100'


 '2-notch7300_1600_30dB.json', 
 '4-notch8300_2000_26dB.json',
 '1-notch7400_bw1000.json',
'2-notch8400_bw1200.json',
'3-notch7800_bw1800.json',
'8-notch8100_bw2000.json',


'1-notch5800_1400_30dB.json', 
'3-notch6500_1400_26dB.json',
'1-notch6600_bw900.json',
'2-notch7000_bw1200.json',

'1-notch4800_1200_30dB.json',
'3-notch5300_1200_26dB.json',
'2-notch5400_bw1000.json',
'3-notch5000_bw800.json'
]


vbw_6k_fln_list=['1-notch5900_bw900.json',
'2-notch6200_bw1000.json',
'3-notch6000_bw1100.json',
'4-notch6100_bw1200.json',
'7-notch5900_bw1300.json',
'8-notch5800_bw1400.json',
'9-notch6300_bw1100.json']


vbw_6k_fln_list+=['1-notch5800_1400_30dB.json',
'3-notch6500_1400_26dB.json',
'5-notch7000_1600_26dB.json',
'1-notch6600_bw900.json',
'2-notch7000_bw1200.json']


# 5k

ntch_5k_masker_list=[
'4_notch5000_bw2000_45dB',
'5_notch5000_bw2000_40dB',
'6_notch5000_bw2000_35dB',
'7_notch5000_bw2000_32dB',
'8_notch5000_bw2000_29dB',
'9_notch5000_bw2000_26dB',
'10_notch5000_bw2000_23dB',
'11_notch5000_bw2000_17dB', #new ones
'1-notch4800_1200_30dB.json',
'2-notch4800_1200_25dB.json',
'3-notch5300_1200_26dB.json'] 


attns_5k=np.array([45,40,35,32,29,26,23, 17])

ntch_5k_re='.*notch5000_bw2000'


vfreq_5k_fln_list=[
'6_notch6000_bw2000_35dB', 
'9_notch6000_bw2000_26dB',


'1-notch7600_bw1100', '2-notch7800_bw1300',
'3-notch8000_bw1400',
'6_notch8000_bw2300_35dB', '8_notch8000_bw2300_26dB',

'1-notch5900_bw900',
'2-notch6200_bw1000',
'8-notch5800_bw1400',

'1-notch4800_bw900.json',
'3-notch5100_bw1200.json',
'4-notch4800_bw1400.json',
'5-notch5100_bw1100.json',
'6-notch4900_bw1300.json',
'7-notch5200_bw900.json'
'6_notch5000_bw2000_35dB',
'9_notch5000_bw2000_26dB',


'6_notch4000_bw1700_35dB',
'9_notch4000_bw1700_26dB',
 '1-notch3800_bw800.json',
'2-notch4000_bw900.json',
'3-notch4000_bw1000.json',
'4-notch4200_bw1100.json',

'1-notch5800_1400_30dB.json', 
'3-notch6500_1400_26dB.json',
'1-notch6600_bw900.json',
'2-notch7000_bw1200.json',

'1-notch4800_1200_30dB.json',
'3-notch5300_1200_26dB.json',
'2-notch5400_bw1000.json',
'3-notch5000_bw800.json',

'1-notch3800_1100_30dB.json',
'3-notch4300_1100_26dB.json',
'1-notch4500_bw700.json']

vbw_5k_fln_list=['1-notch4800_bw900.json',
'3-notch5100_bw1200.json',
'4-notch4800_bw1400.json',
'5-notch5100_bw1100.json',
'6-notch4900_bw1300.json',
'7-notch5200_bw900.json']

vbw_5k_fln_list+=['1-notch4800_1200_30dB.json',
'3-notch5300_1200_26dB.json',
'2-notch5400_bw1000.json',
'3-notch5000_bw800.json']

#4k

ntch_4k_masker_list=[
'4_notch4000_bw1700_45dB',
'5_notch4000_bw1700_40dB',
'6_notch4000_bw1700_35dB',
'7_notch4000_bw1700_32dB',
'8_notch4000_bw1700_29dB',
'9_notch4000_bw1700_26dB',
'10_notch4000_bw1700_23dB',
'11_notch400_bw1700_17dB', #XXX typo
'1-notch3800_1100_30dB.json', 
'2-notch3800_1100_25dB.json',
'3-notch4300_1100_26dB.json',
'4-notch4300_1100_23dB.json']


attns_4k=np.array([45,40,35,32,29,26,23, 17])

ntch_4k_re='(.*notch4000_bw1700)|(.*notch400_bw1700)'

vfreq_4k_fln_list=[

'6_notch6000_bw2000_35dB', 
'9_notch6000_bw2000_26dB',

'1-notch7600_bw1100',

'1-notch5900_bw900',
'2-notch6200_bw1000',
'8-notch5800_bw1400',

'1-notch4800_bw900.json',
'3-notch5100_bw1200.json',
'4-notch4800_bw1400.json',
'5-notch5100_bw1100.json',
'6-notch4900_bw1300.json',
'7-notch5200_bw900.json',

'6_notch5000_bw2000_35dB',
'9_notch5000_bw2000_26dB',


'6_notch4000_bw1700_35dB',
'9_notch4000_bw1700_26dB',
 '1-notch3800_bw800.json',
'2-notch4000_bw900.json',
'3-notch4000_bw1000.json',
'4-notch4200_bw1100.json',
'5-notch4200_bw800.json',
'6-notch3800_bw1200.json',

'7_notch3000_bw1500_35dB'
,'10_notch3000_bw1500_26dB',
'1-notch2900_bw600.json',
'2-notch3000_bw700.json',
'3-notch3300_bw700.json',
'4-notch3100_bw800.json',
'6-notch3000_bw900.json',
'7-notch3100_bw1000.json',

'6_notch2200_bw1500_35dB',
'9_notch2200_bw1500_26dB',
'3-notch2600_bw700.json',
'5-notch2400_bw900.json',

'1-notch4800_1200_30dB.json',
'3-notch5300_1200_26dB.json',
'2-notch5400_bw1000.json',
'3-notch5000_bw800.json',

'1-notch3800_1100_30dB.json',
'3-notch4300_1100_26dB.json',
'1-notch4500_bw700.json',
'1-notch2600_900_30dB.json',
'3-notch3400_900_26dB.json',
'1-notch2800_bw800.json',
'2-notch3400_bw900.json']

vbw_4k_fln_list=['1-notch3800_bw800.json',
'2-notch4000_bw900.json',
'3-notch4000_bw1000.json',
'4-notch4200_bw1100.json',
'5-notch4200_bw800.json',
'6-notch3800_bw1200.json']

vbw_4k_fln_list+=['1-notch3800_1100_30dB.json',
'3-notch4300_1100_26dB.json',
'1-notch4500_bw700.json']


#3k

ntch_3k_masker_list=[
'5_notch3000_bw1500_45dB'
,'6_notch3000_bw1500_40dB'
,'7_notch3000_bw1500_35dB'
,'8_notch3000_bw1500_32dB'
,'9_notch3000_bw1500_29dB'
,'10_notch3000_bw1500_26dB'
,'11_notch3000_bw1500_23dB',
 '12_notch300_bw1500_17dB', #XXX typo
 '1-notch2600_900_30dB.json', #new ones
'2-notch2600_900_25dB.json',
'3-notch3400_900_26dB.json',
'4-notch3400_900_23dB.json']


attns_3k=np.array([45,40,35,32,29,26,23, 17])

ntch_3k_re='(.*notch3000_bw1500)|(.*notch300_bw1500)'

vbw_3k_fln_list=['1-notch2900_bw600.json',
'2-notch3000_bw700.json',
'3-notch3300_bw700.json',
'4-notch3100_bw800.json',
'6-notch3000_bw900.json',
'7-notch3100_bw1000.json']


vbw_3k_fln_list+=['1-notch2600_900_30dB.json',
'3-notch3400_900_26dB.json',
'1-notch2800_bw800.json',
'2-notch3400_bw900.json']


vfreq_3k_fln_list=[

'8-notch5800_bw1400',

'1-notch4800_bw900.json',
'3-notch5100_bw1200.json',
'4-notch4800_bw1400.json',
'5-notch5100_bw1100.json',

'6_notch5000_bw2000_35dB',
'9_notch5000_bw2000_26dB',

'6_notch4000_bw1700_35dB',
'9_notch4000_bw1700_26dB',
 '1-notch3800_bw800.json',
'2-notch4000_bw900.json',
'3-notch4000_bw1000.json',
'4-notch4200_bw1100.json',
'5-notch4200_bw800.json',
'6-notch3800_bw1200.json',

'7_notch3000_bw1500_35dB'
,'10_notch3000_bw1500_26dB',
'1-notch2900_bw600.json',
'2-notch3000_bw700.json',
'3-notch3300_bw700.json',
'4-notch3100_bw800.json',
'6-notch3000_bw900.json',
'7-notch3100_bw1000.json',

'6_notch2200_bw1500_35dB',
'9_notch2200_bw1500_26dB',
'3-notch2600_bw700.json',
'5-notch2400_bw900.json',

'6_notch2200_bw1500_35dB',
'9_notch2200_bw1500_26dB',


'35_notch1500_bw1000_35dB',
'38_notch1500_bw1000_26dB',

'1-notch_1400_bw400.json',
'2-notch1500_bw500.json',
'3-notch1500_bw700.json',
'4-notch1600_bw600.json',
'5-notch1700_bw800.json',

'1-notch3800_1100_30dB.json',
'3-notch4300_1100_26dB.json',
'1-notch4500_bw700.json',

'1-notch2600_900_30dB.json',
'3-notch3400_900_26dB.json',
'1-notch2800_bw800.json',
'2-notch3400_bw900.json',
'3-notch2600_bw700.json',

'3-notch1300_bw400.json',
'4-notch1800_bw500.json'  
]



# 2.2khz

ntch_2200_masker_list=[
'4_notch2200_bw1500_45dB',
'5_notch2200_bw1500_40dB',
'6_notch2200_bw1500_35dB',
'7_notch2200_bw1500_32dB',
'8_notch2200_bw1500_29dB',
'9_notch2200_bw1500_26dB',
'10_notch2200_bw1500_23dB',
'11_notch2200_bw1500_17dB' #new one
]


attns_2200=np.array([45,40,35,32,29,26,23, 17])

ntch_2200_re='.*notch2200_bw1500'


vbw_2200_fln_list=['1-notch2000_bw600.json', 
'2-notch2200_bw800.json',
'3-notch2300_bw900.json',
'3-notch2600_bw700.json', 
'5-notch2400_bw900.json',
'3-notch2600_bw700.json' 
]



# 1.5 kHz


ntch_1500_masker_list=[
'33_notch1500_bw1000_45dB',
'34_notch1500_bw1000_40dB',
'35_notch1500_bw1000_35dB',
'36_notch1500_bw1000_32dB',
'37_notch1500_bw1000_29dB',
'38_notch1500_bw1000_26dB',
'39_notch1500_bw1000_23dB',
'40_notch1500_bw1000_17dB' 
]


attns_1500=np.array([45,40,35,32,29,26,23, 17])

ntch_1500_re='.*notch1500_bw1000'

vbw_1500_fln_list=['1-notch_1400_bw400.json',
'2-notch1500_bw500.json',
'3-notch1500_bw700.json',
'4-notch1600_bw600.json',
'5-notch1700_bw800.json',
'3-notch1300_bw400.json',
'4-notch1800_bw500.json' 
]

# dictionnaries

ntch_masker_lists={8000:ntch_8k_masker_list, 6000:ntch_6k_masker_list,
 5000:ntch_5k_masker_list, 4000:ntch_4k_masker_list, 3000:ntch_3k_masker_list, 
 2200:ntch_2200_masker_list, 1500:ntch_1500_masker_list, }

ntch_regexps={8000:ntch_8k_re, 6000:ntch_6k_re, 5000:ntch_5k_re, 4000:ntch_4k_re,
 3000:ntch_3k_re, 2200:ntch_2200_re, 1500:ntch_1500_re}   

vfreq_fln_lists={8000:vfreq_8k_fln_list, 6000:vfreq_6k_fln_list, 
   5000:vfreq_5k_fln_list, 4000:vfreq_4k_fln_list,
    3000:vfreq_3k_fln_list, 2200:vfreq_3k_fln_list, 1500:vfreq_3k_fln_list}
     


vbw_fln_lists={8000:vbw_8k_fln_list, 6000:vbw_6k_fln_list, 5000:vbw_5k_fln_list, 4000:vbw_4k_fln_list,
 3000:vbw_3k_fln_list, 2200: vbw_2200_fln_list, 1500:vbw_1500_fln_list }

attns_arrays={
8000:attns_8k,
6000:attns_6k,
5000:attns_5k,
4000:attns_4k,
3000:attns_3k,
2200:attns_2200,
1500:attns_1500}

