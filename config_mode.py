#util module to share CAP mode (C/R/C+R) accross scripts

mode=None

def init(mode_str):
	assert mode_str in ['C', 'R', 'C+R']
	global mode
	mode = mode_str


