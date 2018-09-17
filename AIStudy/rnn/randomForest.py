
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
from sklearn.tree import export_graphviz
from sklearn import tree
import pydotplus 

score_init = 0
model_init = ''
score_dict = {}


def load_data():
	data = pd.read_csv('data.csv')
	data.dropna(inplace=True)
	X = data[["R001_014","R001_016","R001_018","R001_020","R001_022","R001_013","R001_015","R001_017","R001_019","R001_021"]]
	Y = data ['RATE']
	X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=12)
	return X_train,X_test,Y_train,Y_test


def try_different_method(model):
	global score_dict
	x_train, x_test, y_train, y_test = load_data()
	t0 = time.time()
	model.fit(x_train,y_train)
	print(model.__class__.__name__+' time is : '+str(time.time()-t0))
	dot_data = tree.export_graphviz(model,out_file=None)
	graph = pydotplus.graph_from_dot_data(dot_data)
	graph.write_pdf("ppp.pdf")

    

def create_model():
	dict = {} 
	model_DecisionTreeRegressor = tree.DecisionTreeRegressor() 
	dict['DecisionTreeRegressor'] = model_DecisionTreeRegressor
	return dict

def use_model():
	global score_dict
	dict = {}
	dict_model = create_model()
	for i in range(1):
		for key in dict_model:
			try_different_method(dict_model[key])
	print("done")
    
use_model()
