import numpy as np
import csv

from latencies import PowerLawLatencies

class URfromCsv:

	def __init__(self,filename, name=''):

		'''
		Args:
			filename (str) : csv filename. 2 fields: time (in ms), amplitude
		'''
		with open(filename, 'r') as f:
			csv_reader=csv.DictReader(f, delimiter=',')
			t=[]
			u=[]

			#
			for column in csv_reader.fieldnames:
				if 'time' in column.lower():
					tCol = column 
				if 'amplitude' in column.lower():
					uCol=column

			for row in csv_reader:
				t.append(float(row[tCol])*1e-3)
				u.append(float(row[uCol]))



			t,u= np.array(t), np.array(u)

			ind=np.argsort(t)
			self._t, self._u = t[ind], u[ind]
		self.name=name

	def u(self, t):
		'''
		Args:
			t (float or numpy array)
		Returns:
			float or numpy array: UR for t
		'''
		return np.interp(t, self._t, self._u)



URWang1979 = URfromCsv('./UR/Wang1979Fig14.csv', name='averaged UR (Wang 1979, Fig14)')
Eggermont1976clickLatencies80dB=PowerLawLatencies.fromPts(5.3e-3, 1e3, 2e-3, 5e3, name="Eggermont 1976 click 80dB")