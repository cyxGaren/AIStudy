import pandas as pd
import pymysql
from sqlalchemy import create_engine

engine = create_engine("mysql+pymysql://root:123456@127.0.0.1:3306/cyx?charset=utf8") 

def load_data(filenames):
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
	return data


data=load_data(["rrc.csv"])

data.to_sql(name='rrc',con=engine,if_exists='append',index=False,index_label=False)
