from numpy import *
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

step = 10

input_size = 1
output_size = 1
rnn_unit = 5
batch_size = 20

#读取数据
def load_data():
	data = pd.read_csv('data.csv')
	#已按时间序列排好
	data = array(data['RATE']).reshape([-1,input_size])
	x,y =[],[]
	scaler = preprocessing.StandardScaler().fit(data)
	data = scaler.transform(data)
	for i in range(len(data)-step):
		x.append(data[i:i+step].tolist())
		y.append(data[i+step].tolist())
	return array(x).reshape([-1,step,input_size]),array(y).reshape([-1,1]),scaler

X = tf.placeholder('float32',[None,step,input_size])
Y = tf.placeholder('float32',[None,output_size])

weight = {
	'in':	tf.Variable(tf.random_normal([input_size,rnn_unit])),
	'out':	tf.Variable(tf.random_normal([rnn_unit,output_size]))
}
baies = {
	'in':	tf.Variable(tf.random_normal([rnn_unit])),
	'out':	tf.Variable(tf.random_normal([output_size]))
}

def lstm(batch):
	w_in = weight['in']
	w_out = weight['out']
	b_in = baies['in']
	b_out = baies['out']
	#转型计算
	input_data = tf.reshape(X,[-1,input_size])
	rnn_data = tf.add(tf.matmul(input_data,w_in),b_in)
	rnn_data = tf.reshape(rnn_data,[-1,step,rnn_unit])
	#创建神经元
	cell =	tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
	h0 = cell.zero_state(batch,'float32')
	output_data,last_data = tf.nn.dynamic_rnn(cell,rnn_data,initial_state=h0)
	last_data = tf.reshape(last_data.h,[-1,rnn_unit])
	pred_y = tf.add(tf.matmul(last_data,w_out),b_out)
	return pred_y

def train():
	pred = lstm(batch_size)
	loss = tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y,[-1])))
	optimizer = tf.train.AdamOptimizer(0.006)
	train_op = optimizer.minimize(loss)
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(init)
		for i in range(10000):
			start = 0
			end = start+batch_size
			while end<len(train_y):
				_,_loss = sess.run([train_op,loss],feed_dict={X:train_x[start:end],Y:train_y[start:end]})
				start = end
				end += batch_size	
			saver.save(sess,'ckpt/8-6.ckpt')
			if i+1 % 500 ==0:
				print(i/100,"%") 


def load():
	pred = lstm(1)
	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess,'ckpt/8-6.ckpt')
		yy = []
		for i in range(len(train_y)):
			yy.extend(sess.run(pred,feed_dict={X:train_x[i:i+1]}))
		yy = array(yy).reshape([-1])
		yy = scaler.inverse_transform(yy)
		y = scaler.inverse_transform(train_y)
		plt.plot(y)
		plt.plot(yy)
		plt.legend(['no1','no2'])
		plt.savefig('pic/lstm.png')

train_x,train_y,scaler = load_data()
#train()
load()

#load_data()






