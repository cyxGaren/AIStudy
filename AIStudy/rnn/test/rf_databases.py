
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
import pymysql
from sqlalchemy import create_engine

engine = create_engine("mysql+pymysql://root:123456@127.0.0.1:3306/cyx?charset=utf8")

class RandomForest:	
	def StartForestTrain(self,modelname,days):
		X,Y = self.load_data(days)
		#self.create_model(modelname,X,Y)
	
	def UseForestTrain(self,modelname,days):
		model = self.get_model(modelname)
		for filename in filenames:
			self.use_model(model,filename)
			
	def load_data(self,days):
		sql1 = 'select BEGIN_TIME - interval '+days+' day from rrc order by BEGIN_TIME desc limit 1'
		sql2 = 'select BEGIN_TIME + interval 0 day from rrc order by BEGIN_TIME desc limit 1'
		date1 = pd.read_sql(sql1,con=engine)
		date2 = pd.read_sql(sql2,con=engine)
		sql3 = "select * from rrc where BEGIN_TIME between '"+date1.iat[0,0]+"' and '"+date2.iat[0,0]+"'"
		data = pd.read_sql(sql3,con=engine)
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
		sql1 = 'select BEGIN_TIME - interval '+days+' day from rrc order by BEGIN_TIME desc limit 1'
		sql2 = 'select BEGIN_TIME + interval 0 day from rrc order by BEGIN_TIME desc limit 1'
		date1 = pd.read_sql(sql1,con=engine)
		date2 = pd.read_sql(sql2,con=engine)
		sql = "select * from rrc where BEGIN_TIME between '"+date1.iat[0,0]+"' and '"+date2.iat[0,0]+"'"
		data = pd.read_sql(sql,con=engine)
		data.dropna(inplace=True)
		X = data[["R001_014","R001_016","R001_018","R001_020","R001_022","R001_013","R001_015","R001_017","R001_019","R001_021"]] 
		result = model.predict(X)
		print(result)
		

rf=RandomForest()
modelname = sys.argv[2]
days = sys.argv[3]
if sys.argv[1]=='train':
	rf.StartForestTrain(modelname,days)
elif sys.argv[1]=='pred':
	rf.UseForestTrain(modelname,days)
else:
	print('wrong args')
