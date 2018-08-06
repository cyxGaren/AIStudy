import tensorflow as tf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy import *
from sklearn import preprocessing

step = 10	#以10为循环大小
batch_size = 30	#每次预测30

input_size = 1
output_size = 1
rnn_unit = 5



#读取csv数据(已按时间分离)
def load_data():
	data_rate = pd.read_csv('data.csv')['RATE']
	#标准化,减少误差
	data_rate=array(data_rate).reshape([-1,1])
	scaler = preprocessing.StandardScaler().fit(data_rate)
	data_rate = scaler.transform(data_rate)
	x,y = [],[]
	for i in range(len(data_rate)-step):
		x.append(data_rate[i:i+step])
		y.append(data_rate[i+1])
	return array(x).reshape([-1,step,input_size]),array(y).reshape([-1,output_size]),scaler

###################神经网络######################

weight = {
	'in':	tf.Variable(tf.random_normal([input_size,rnn_unit])),
	'out':	tf.Variable(tf.random_normal([rnn_unit,output_size]))
}

baies = {
	'in':	tf.Variable(tf.random_normal([rnn_unit,])),
	'out':	tf.Variable(tf.random_normal([output_size,]))
}

X = tf.placeholder('float32',[None,step,input_size])
Y = tf.placeholder('float32',[None,output_size])

def lstm(batch):
	#3维转2维
	input_data = tf.reshape(X,[-1,input_size])
	rnn_data = tf.add(tf.matmul(input_data,weight['in']),baies['in'])
	rnn_data = tf.reshape(rnn_data,[-1,step,rnn_unit])

	cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
	h0 = cell.zero_state(batch,'float32')
	_,last_data = tf.nn.dynamic_rnn(cell,rnn_data,initial_state=h0)
	last_data = last_data.h
	
	pred_y = tf.add(tf.matmul(last_data,weight['out']),baies['out'])
	return pred_y

###################训练###############################
def train():
	global batch_size
	pred_y = lstm(batch_size)
	loss = tf.reduce_mean(tf.square(tf.reshape(pred_y,[-1])-tf.reshape(Y,[-1])))
	optimizer = tf.train.AdamOptimizer(0.006)
	train_op = optimizer.minimize(loss)
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(init)
		num = 1000
		for i in range(num):
			start = 0
			end = start+batch_size
			while end<len(train_y):
				_loss,_ = sess.run([loss,train_op],feed_dict={X:train_x[start:end],Y:train_y[start:end]})
				start =end
				end = end+batch_size
			if (i+1)%50 == 0:			
				print((i+1)/(num/100),'%')
			saver.save(sess,'../ckpt/rnn.ckpt')

##################预测##################################
def use_model():
	pred_y = lstm(1)
	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess,'../ckpt/rnn.ckpt')
		#训练集最后一行为测试样本
		test_x_list = train_x[-1:] 
		pred_y_list = []
		for i in range(10):
			pred = sess.run(pred_y,feed_dict={X:test_x_list})
			test_x_list[-1] = vstack((test_x_list[-1,1:],pred))
			pred_y_list.extend(pred)
		plt.plot(scaler.inverse_transform(pred_y_list))
		plt.plot(scaler.inverse_transform(test_y))
		plt.legend(['predy','testy'])
		plt.savefig('../pic/pred.png')

x,y,scaler = load_data()
index = (int)(3.0/7*len(x))
train_x,train_y,test_x,test_y = x[:index],y[:index],x[index:],y[index:]
#train()
use_model()
