
from numpy import *
import seaborn as sns
import pandas as pd
from sklearn.linear_model import *
from sklearn.pipeline import *
from sklearn.model_selection import train_test_split
from numpy import float64
from sklearn import *
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.externals import joblib
import time
import sys
import csv

class RandomForest:	
	def StartForestTrain(self,modelname,*filenames):
		X,Y = self.load_data(filenames)
		self.create_model(modelname,X,Y)
	
	def UseForestTrain(self,modelname,*filenames):
		model = self.get_model(modelname)
		for filename in filenames:
			self.use_model(model,filename)
			
	def load_data(self,filenames):
		data = []
		tag = 0
		for filename in filenames:	
			if tag == 0:
				data = pd.read_csv(filename)
				data.dropna(inplace=True)
				tag = 1
			else:
				databranch = pd.read_csv(filename)
				databranch.dropna(inplace=True)
				data=(pd.concat([data,databranch],axis=0))
		X = data[["R001_014","R001_016","R001_018","R001_020","R001_022","R001_013","R001_015","R001_017","R001_019","R001_021"]]
		Y = data ['RATE']
		return X,Y
	
	
	def create_model(self,modelname,X,Y):
		print('start training................................')
		model = ensemble.RandomForestRegressor(n_estimators=20)
		x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=12)
		time_start = time.time()
		model.fit(x_train,y_train)
		print('end training................................')
		print(model.__class__.__name__+' train time: '+str(time.time()-time_start))
		score = model.score(x_test, y_test)
		print(model.__class__.__name__+' score: '+str(score))
		joblib.dump(model, modelname)

	def get_model(self,modelname):
		model = joblib.load(modelname)
		return model
	
	def use_model(self,model,filename):
		data = pd.read_csv(filename)
		data.dropna(inplace=True)
		X = data[["R001_014","R001_016","R001_018","R001_020","R001_022","R001_013","R001_015","R001_017","R001_019","R001_021"]] 
		result = model.predict(X)
		Y = data['RATE']
		csvfile = open('rf_pred.csv','w')
		writer = csv.writer(csvfile)
		writer.writerow(["回归阈值"])
		a = []
		for i in range(90000):
			a.append(result[i])
		writer.writerow(a)
		csvfile.close()
			
			
		

rf=RandomForest()
modelname = sys.argv[2]
filename = sys.argv[3]
if sys.argv[1]=='train':
	rf.StartForestTrain(modelname,filename)
elif sys.argv[1]=='pred':
	rf.UseForestTrain(modelname,filename)
else:
	print('wrong args')
