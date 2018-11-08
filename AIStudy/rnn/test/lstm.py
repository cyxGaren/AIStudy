import os
import time

class LSTM:
	def StartLSTMTrain(self,filename):
		time_start = time.time()
		print('start training.............')
		os.system('python lstm_train.py '+filename+' rate train')
		os.system('python lstm_train.py '+filename+' threshold train')
		print('end training..............')
		print('training time: '+str(time.time()-time_start))

	def UseLSTMModel(self,filename):
		os.system('python lstm_train.py '+filename+' rate pred')
		os.system('python lstm_train.py '+filename+' threshold pred')
		
lstm = LSTM()
lstm.UseLSTMModel('test.csv')
