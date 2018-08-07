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



#创建数据
def load_data():
	random.seed(5)
	x_data = array(range(400))
	y_data = sin(x_data)+sqrt(x_data)*10+random.rand(400)
	y_data = array(y_data).reshape([-1,1])
	scaler = preprocessing.StandardScaler().fit(y_data)
	y_data = scaler.transform(y_data)
	x,y = [],[]
	for i in range(len(y_data)-step):
		x.append(y_data[i:i+step])
		y.append(y_data[i+1])
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
def train(model_name):
	global batch_size
	pred_y = lstm(batch_size)
	loss = tf.reduce_mean(tf.square(tf.reshape(pred_y,[-1])-tf.reshape(Y,[-1])))
	optimizer = tf.train.AdamOptimizer(0.006)
	train_op = optimizer.minimize(loss)
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(init)
		num = 10000
		for i in range(num):
			start = 0
			end = start+batch_size
			while end<len(train_y):
				_loss,_ = sess.run([loss,train_op],feed_dict={X:train_x[start:end],Y:train_y[start:end]})
				start =end
				end = end+batch_size
			if (i+1)%50 == 0:			
				print((i+1)/(num/100),'%')
			saver.save(sess,'../ckpt/'+model_name+'.ckpt')

##################预测##################################
def use_model(model_name,pic_name):
	pred_y = lstm(1)
	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess,'../ckpt/'+model_name+'.ckpt')
		#第一组以 训练集最后一行为测试样本,预测后一天的阈值(即根据前20时间段来估计下一次的阈值)
		#随后将每次生成的预测值插入输入集队列尾(即保持一直使用预测集中的数据)
		
		#第二组以 测试集的输入集合为样本 预测后一天(即使用前20个真实值来预测第21个时间段的阈值)
		test_x_list_1 = train_x[-1:] 
		test_x_list_2 = test_x
		pred_y_list_1 = []
		pred_y_list_2 = []
		for i in range(len(test_y)):
			pred = sess.run(pred_y,feed_dict={X:test_x_list_1})
			test_x_list_1[-1] = vstack((test_x_list_1[-1,1:],pred))
			pred_y_list_1.extend(pred)
			pred_y_list_2.extend(sess.run(pred_y,feed_dict={X:test_x_list_2[i:i+1]}))
		_test_y = scaler.inverse_transform(test_y)
		pred_y_list_1 = scaler.inverse_transform(pred_y_list_1)
		pred_y_list_2 = scaler.inverse_transform(pred_y_list_2)
		_train_y = scaler.inverse_transform(train_y)
		plt.plot(range(0,index),_train_y)
		plt.plot(range(index,_length),_test_y)
		plt.plot(range(index,_length),pred_y_list_1)
		plt.plot(range(index,_length),pred_y_list_2)
		plt.legend(['trainY','testY','allPredY','testxY'])
		plt.savefig('../pic/'+pic_name+'.png')

x,y,scaler = load_data()
_length = len(x)
index = (int)(5.0/7*len(x))
train_x,train_y,test_x,test_y = x[:index],y[:index],x[index:],y[index:]
#train('test')
use_model('test','test')
