
import tensorflow as tf
import pandas as pd
from numpy import *
from sklearn import preprocessing
class Load(object):
	file_name = ''
	step = 20
	input_size = 1
	scaler = None
	def __init__(self,file_name):
		self.file_name = file_name
	
	def load_data(self):
		date_pares_func = lambda x:pd.datetime.strptime(x,'%Y/%m/%d')
		data = pd.read_csv(self.file_name,parse_dates=['date'],index_col=['date'],date_parser=date_pares_func)
		data = data.sort_index(ascending=True)
		data = data.values
	
		#标准化
		data = data.reshape([-1,self.input_size])
		scaler = preprocessing.StandardScaler().fit(data)
		self.scaler = scaler
		data = scaler.transform(data)
		x,y = [],[]
		for i in range(len(data)-self.step):
			x.append(data[i:i+self.step].tolist())
			y.append(data[i+self.step].tolist())
		return array(x),array(y)

	def get_scaler(self):
		return self.scaler
