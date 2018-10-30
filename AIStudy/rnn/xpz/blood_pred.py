import tensorflow as tf
from numpy import *
from sklearn import preprocessing


step = 2
input_size = 1
rnn_unit = 5
output_size = 1

def load_data():
	arrayy = [1526054400000,1529424000000,1531756800000,1533830400000,1536076800000]
	arrayy=array(arrayy).reshape([-1,1])
	scaler = preprocessing.StandardScaler().fit(arrayy)
	arrayy = scaler.transform(arrayy)
	x,y = [],[]
	for i in range(len(arrayy)-step):
		x.append(arrayy[i:i+step])
		y.append(arrayy[i+step])
	x = array(x)
	return x.reshape([-1,step,input_size]),array(y).reshape([-1,output_size]),scaler

weight = {
	'in': tf.Variable(tf.random_normal([input_size,rnn_unit])),
	'out': tf.Variable(tf.random_normal([rnn_unit,output_size]))
}

baies = {
	'in': tf.Variable(tf.random_normal([rnn_unit,])),
	'out': tf.Variable(tf.random_normal([output_size,]))
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
	pred_y = lstm(3)
	loss = tf.reduce_mean(tf.square(tf.reshape(pred_y,[-1])-tf.reshape(Y,[-1])))
	optimizer = tf.train.AdamOptimizer(0.001)
	train_op = optimizer.minimize(loss)
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(init)
		num = 100000
		for i in range(num):
			_loss,_ = sess.run([loss,train_op],feed_dict={X:x[:],Y:y[:]})
			if (i+1)%50 == 0:			
				print((i+1)/(num/100),'%')
			saver.save(sess,'./rnn.ckpt')

##################预测##################################
def use_model():
	pred_y = lstm(1)
	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess,'./rnn.ckpt')
		xx = [1533830400000,1536076800000]
		xx = array(xx).reshape([-1,step,input_size])
		pred = sess.run(pred_y,feed_dict={X:xx})
		print('%f'%scaler.inverse_transform(pred))
x,y,scaler = load_data()
#train()
use_model()


