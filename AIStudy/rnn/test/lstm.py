import os
import time

class LSTM:
	def StartLSTMTrain(self,filename):
		time_start = time.time()
		print('start training.............')
		os.system('python lstm_train.py '+filename+' rate')
		os.system('python lstm_train.py '+filename+' threshold')
		print('end training..............')
		print('training time: '+str(time.time()-time_start))

lstm = LSTM()
lstm.StartLSTMTrain('test.csv')
