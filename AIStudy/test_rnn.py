from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from sklearn import preprocessing
from numpy import *
	

'''
定义神经网络
每次输入为	数量*步长*维度	即将步长*维度看做一组输入
W_in	[维度*隐层]
维度为1:	1*W+b	1*隐层
维度为n:	n*W+b	n*隐层
步长为m:	m(n*W+b)	m(n*隐层)
三维转二维 mult	然后二维转三维 add
'''
x = array(range(400))
x = [float32(y) for (y) in x]
#y = np.sin(x)+np.random.rand(400)*0.5+np.sqrt(x)*0.1

def rnn_format_data(data,step):
	x = []
	y = []	
	for i in range(len(data)-step):
		y.append(data[i+step])
	for i in range(len(data)-step):
		x.append(data[i:(i+step)])
	return x,y

input_n = 1	#输入维度
step = 5
rnn_unit = 6	#隐层的大小
output_n = 1	#输出维度
lr = 0.006


x,y = rnn_format_data(x,step)
x = array(x).reshape(-1,step,input_n) #因为rnn需要输入为 batch*step*inputsize


#定义神经网络变量
weight = {
	'in':tf.Variable(tf.random_normal([input_n,rnn_unit])),
	'out':tf.Variable(tf.random_normal([rnn_unit,1]))
}

biases = {
	'in':tf.Variable(tf.random_normal([rnn_unit,])),
	'out':tf.Variable(tf.random_normal([output_n,]))
}
x_data = tf.placeholder('float',[None,step,input_n])
y_data = tf.placeholder('float',[None,step,output_n])

#定义神经网络:
def lstm(data):
	global step,rnn_unit
	w_in = weight['in']
	b_in = biases['in']
	data = tf.reshape(data,[-1,1])
	rnn_hidden = tf.matmul(data,w_in)+b_in	
	rnn_hidden = tf.reshape(rnn_hidden,[-1,step,rnn_unit])
	cell = tf.nn.rnn_cell.BasicRNNCell(rnn_unit) #一层隐层的神经网络
	init_state = cell.zero_state(395,dtype='float32')
	out,final_val = tf.nn.dynamic_rnn(cell,rnn_hidden,initial_state=init_state)
	out = tf.reshape(out,[-1,rnn_unit])
	w_out = weight['out']
	b_out = biases['out']
	y = tf.matmul(out,w_out)+b_out
	print(y.shape)
	return y

def train():
	global x,y
	y_train = lstm(x)
	loss = tf.reduce_mean(tf.square(tf.reshape(y_train,[-1])-tf.reshape(y,[-1])))
	optimizer = tf.train.AdamOptimizer(lr)
	train_ = optimizer.minimize(loss)
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		for i in range(1000):
			_loss,_train = sess.run([train_,loss])
			print(_loss)


train()



