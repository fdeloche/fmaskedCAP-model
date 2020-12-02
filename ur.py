import numpy as np
import csv
import copy

class URfromCsv:

	def __init__(self,filename, name='', shifted=False):

		'''
		Args:
			filename (str) : csv filename. 2 fields: time (in ms), amplitude
			shifted (boolean) : if True, time array begins at 0
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
			if shifted:
				self._t-=t[0]
		self._func = lambda x: 1
		self.name=name

	@classmethod
	def modify(cls, ur, func, name=None):
		'''
		Returns a new UR based on ur with a function (of t). u will be initial ur multiplied by func. Based on shallow copy
		''' 
		ur2=copy.copy(ur)
		ur2._func=func
		if name is not None:
			ur2.name=name
		else:
			ur2.name = ur2.name + " modified"
		return  ur2


	def u(self, t):
		'''
		Args:
			t (float or numpy array)
		Returns:
			float or numpy array: UR for t
		'''
		return np.interp(t, self._t, self._u, left=0., right=0.)*self._func(t)

	def __call__(self, t):
		return self.u(t)

URWang1979 = URfromCsv('./UR/Wang1979Fig14.csv', name='averaged UR (Wang 1979, Fig14)')
URWang1979m = URfromCsv.modify(URWang1979, lambda t:1+4*np.exp(-1/2*(t+0.2e-4)**2/1e-4**2)) #produces more realistic CAP

URWang1979shifted = URfromCsv('./UR/Wang1979Fig14.csv', name='averaged UR (Wang 1979, Fig14)', shifted=True)

