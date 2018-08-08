import load
import tensorflow as tf


l = load.Load('../dataset_1.csv')

x,y = l.load_data()
scaler = l.get_scaler()

step = l.step
input_size = l.input_size
batch_size = 30
output_size = 1
rnn_unit = 5

num = 1

index = (int)((5.0/7.0)*len(x))
train_x,train_y = x[:index],y[:index]


###################################### RNN variables ##############################
weight = {
	'in':	tf.Variable(tf.random_normal([input_size,rnn_unit]),name='w_in'),
	'out':	tf.Variable(tf.random_normal([rnn_unit,output_size]),name='w_out')
}

baies = {
	'in':	tf.Variable(tf.random_normal([rnn_unit,]),name='b_in'),
	'out':	tf.Variable(tf.random_normal([output_size,]),name='b_out')
}

X = tf.placeholder('float32',[None,step,input_size],name='X')
Y = tf.placeholder('float32',[None,output_size],name='Y')


###################################### RNN func ###################################
def lstm(batch):
	rnn_data = tf.add(tf.matmul(tf.reshape(X,[-1,input_size]),weight['in']),baies['in'])
	rnn_data = tf.reshape(rnn_data,[-1,step,rnn_unit])
	cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
	h0 = cell.zero_state(batch,'float32')
	_,last = tf.nn.dynamic_rnn(cell,rnn_data,initial_state=h0)
	last = last.h
	output_data = tf.add(tf.matmul(last,weight['out']),baies['out'])
	return output_data
	
####################################

def train():
	pred_y = lstm(batch_size)
	loss = tf.reduce_mean((tf.square(tf.reshape(Y,[-1])-tf.reshape(pred_y,[-1]))))
	optimizer = tf.train.AdamOptimizer(0.006)
	train_op = optimizer.minimize(loss)
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(init)
		for i in range(num):
			start = 0
			end = start+batch_size
			while(end<len(train_y)):
				sess.run(train_op,feed_dict={X:train_x[start:end],Y:train_y[start:end]})
				start = end
				end += batch_size
			if i==0 or (i+1)%50 == 0:
				print((i+1)/num*100,'%')
				if i==0 or (i+1)/num*100%10 == 0:
					saver.save(sess,'/ckpt/rnn.ckpt')
train()
