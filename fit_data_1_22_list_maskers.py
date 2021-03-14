#8k

ntch_8k_masker_list=['2_notch8000_bw2300_55dB',
 '3_notch8000_bw2300_50dB',
 '4_notch8000_bw2300_45dB',
 '5_notch8000_bw2300_40dB',
 '6_notch8000_bw2300_35dB',
 '7_notch8000_bw2300_32dB',
 '7_notch8000_bw2300_29dB',
 '8_notch8000_bw2300_26dB',
 '9_notch8000_bw2300_23dB']

ntch_8k_re='.*notch8000_bw2300'


vfreq_8k_fln_list=['1_hp_10000Hz', '2_hp_9000Hz', '3_hp_8000Hz', '4_hp_7000Hz', '5_hp_6000Hz', '6_hp_5000Hz',
           '6_notch8000_bw2300_35dB', '8_notch8000_bw2300_26dB',
'1-notch7600_bw1100.json', '2-notch7800_bw1300.json',
'3-notch8000_bw1400.json',
'4-notch8200_bw1300.json',
'5-notch8200_bw1500.json',
'6-notch7900_bw1600.json',
'7-notch8100_bw1200.json',         
'6_notch6000_bw2000_35dB', '9_notch6000_bw2000_26dB',
'2-notch6200_bw1000.json', '3-notch6000_bw1100.json', '9-notch6300_bw1100.json'
]

vbw_8k_fln_list=['1-notch7600_bw1100.json',
'2-notch7800_bw1300.json',
'3-notch8000_bw1400.json',
'4-notch8200_bw1300.json',
'5-notch8200_bw1500.json',
'6-notch7900_bw1600.json',
'7-notch8100_bw1200.json']



# 6k

ntch_6k_masker_list=['2_notch6000_bw2000_55dB',
'3_notch6000_bw2000_50dB',
'4_notch6000_bw2000_45dB',
'5_notch6000_bw2000_40dB',
'6_notch6000_bw2000_35dB',
'7_notch6000_bw2000_32dB',
'8_notch6000_bw2000_29dB',
'9_notch6000_bw2000_26dB',
'10_notch6000_bw2000_23dB']

ntch_6k_re='.*notch6000_bw2000'

vfreq_6k_fln_list=['2_hp_9000Hz', '3_hp_8000Hz', '4_hp_7000Hz', '5_hp_6000Hz', '6_hp_5000Hz',
'7_hp_4000Hz', '8_hp_3200Hz', 
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
]


vbw_6k_fln_list=['1-notch5900_bw900.json',
'2-notch6200_bw1000.json',
'3-notch6000_bw1100.json',
'4-notch6100_bw1200.json',
'7-notch5900_bw1300.json',
'8-notch5800_bw1400.json',
'9-notch6300_bw1100.json']

# 5k

ntch_5k_masker_list=['2_notch5000_bw2000_55dB',
'3_notch5000_bw2000_50dB',
'4_notch5000_bw2000_45dB',
'5_notch5000_bw2000_40dB',
'6_notch5000_bw2000_35dB',
'7_notch5000_bw2000_32dB',
'8_notch5000_bw2000_29dB',
'9_notch5000_bw2000_26dB',
'10_notch5000_bw2000_23dB'] 

vbw_5k_fln_list=['1-notch4800_bw900.json',
'2-notch5000_bw1000.json',
'3-notch5100_bw1200.json',
'4-notch4800_bw1400.json',
'5-notch5100_bw1100.json',
'6-notch4900_bw1300.json',
'7-notch5200_bw900.json']

# dictionnaries

ntch_masker_lists={8000:ntch_8k_masker_list, 6000:ntch_6k_masker_list}

ntch_regexps={8000:ntch_8k_re, 6000:ntch_6k_re}

vfreq_fln_lists={8000:vfreq_8k_fln_list, 6000:vfreq_6k_fln_list}


vbw_fln_lists={8000:vbw_8k_fln_list, 6000:vbw_6k_fln_list}


