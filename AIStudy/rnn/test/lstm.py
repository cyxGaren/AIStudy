import os
import time
import pickle
import pandas as pd
import numpy as np
import sys

step=10

class LSTM:
	def StartLSTMTrain(self,filename):
		time_start = time.time()
		print('start training.............')
		os.system('python lstm_train_pred.py '+filename+' rate train')
		os.system('python lstm_train_pred.py '+filename+' threshold train')
		print('end training..............')
		print('training time: '+str(time.time()-time_start))

	def UseLSTMModel(self,filename):
		os.system('python lstm_train_pred.py '+filename+' rate pred')
		os.system('python lstm_train_pred.py '+filename+' threshold pred')
		f1=open('./ckpt/pred_rate.txt','rb')
		f2=open('./ckpt/pred_threshold.txt','rb')                
		rate=pickle.loads(f1.read())
		threshold=pickle.loads(f2.read())
		alarm=[]
		for i in range(len(rate)):
			if (rate[i]-threshold[i])<0:
				alarm.append(1)
			else:
				alarm.append(0)
		data = pd.read_csv('test.csv')['ALARM']
		index = (int)(3.0/7*(data.shape[0]-step))+1
		data=data[index:]
		true_pred=0
		for i in range(len(alarm)):
			if alarm[i]==data.loc[index+i]:
				true_pred=true_pred+1
		print(true_pred/len(alarm))

lstm = LSTM()
filename = sys.argv[2]
if sys.argv[1]=='train':
	lstm.StartLSTMTrain(filename)
elif sys.argv[1]=='pred':
	lstm.UseLSTMModel(filename)
else:
	print('wrong args')
