import pandas as pd
from numpy import *
import sys
import pickle
import sklearn
from sklearn import preprocessing


class Load_data:
	arga = {}
	def __init__(self,input_size,output_size,step,file_name):
		self.arga['input_size'] = input_size
		self.arga['output_size'] = output_size
		self.arga['step'] = step
		self.arga['file_name'] = file_name
	
	def create_data(self,is_init=0):
		#时间lambda表达式
		dateparser = lambda X:pd.datetime.strptime(X,'%Y/%m/%d')
		data = pd.read_csv(self.arga['file_name'],parse_dates=['date'],index_col=['date'],date_parser=dateparser)
		data.sort_index(ascending=True)
		x = data['price']
		y = data['price']
		x = array(x).reshape([-1,self.arga['input_size']])
		y = array(y).reshape([-1,self.arga['output_size']])
		scaler = {}

		##### 	set/get scaler 	#####
		if is_init != 0:
			scaler['x'] = preprocessing.StandardScaler().fit(x)
			scaler['y'] = preprocessing.StandardScaler().fit(y)
			f = open('data/scaler.pkl','wb')
			pickle.dump(scaler,f)
		else:
			f = open('data/scaler.pkl','rb')
			scaler = pickle.load(f)
		f.close()
		
		#####	正则化		#####
		x = scaler['x'].transform(x)
		y = scaler['y'].transform(y)
		return x,y,scaler
		
	def get_args(self):
		print('args is ',self.arga)
	

		
