import tensorflow as tf
import load_data
from numpy import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



l = load_data.Load('../dataset_1.csv')
x,y = l.load_data()
scaler = l.get_scaler()


step = l.step
input_size = l.input_size
batch_size = 1
output_size = 1
rnn_unit = 5

num = 1
        
x_pred = x[0]
x_pred = x_pred.reshape([-1,step,input_size])

weight = {
	'in':   tf.Variable(tf.random_normal([input_size,rnn_unit]),name='w_in'),
	'out':  tf.Variable(tf.random_normal([rnn_unit,output_size]),name='w_out')
}       
        
baies = {
	'in':   tf.Variable(tf.random_normal([rnn_unit,]),name='b_in'),
	'out':  tf.Variable(tf.random_normal([output_size,]),name='b_out')
} 

X = tf.placeholder('float32',[None,step,input_size],name='X')
Y = tf.placeholder('float32',[None,output_size],name='Y')


def lstm(batch):
	rnn_data = tf.add(tf.matmul(tf.reshape(X,[-1,input_size]),weight['in']),baies['in'])
	rnn_data = tf.reshape(rnn_data,[-1,step,rnn_unit])
	cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
	h0 = cell.zero_state(batch,'float32')
	_,last = tf.nn.dynamic_rnn(cell,rnn_data,initial_state=h0)
	last_val = last[-1]
	out = []
	out.append(last_val)
	out.append(_[-1])
	output_data = tf.add(tf.matmul(last_val,weight['out']),baies['out'])
	return out,output_data

##############################预测##################
def pred():
	rnn,pred_y = lstm(1)	
	saver = tf.train.Saver()
	global x_pred
	with tf.Session() as sess:
		saver.restore(sess,'/ckpt/rnn.ckpt')
		predy_by_x_list = []
		predy_by_y_list = []
		for i in range(len(y)):
			predy_by_x_list.append(sess.run(pred_y,feed_dict={X:x[i:i+1]}))
			pred_by_y = sess.run(pred_y,feed_dict={X:x_pred})
			predy_by_y_list.append(pred_by_y)
			x_pred[0] = vstack((x_pred[0][1:],pred_by_y))
		plt.plot(scaler.inverse_transform(y))
		plt.plot(scaler.inverse_transform(predy_by_x_list).reshape([-1]))
		plt.plot(scaler.inverse_transform(predy_by_y_list).reshape([-1]))
		plt.legend(['Y','xY','yY'])
		plt.savefig('demo.png')

pred()
