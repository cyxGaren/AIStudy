from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

mnist=input_data.read_data_sets('/tmp/data/',one_hot=True)


n_hidden_1 = 256	# num of 1st hidder
n_hidden_2 = 256	
n_input = 784
n_output = 10

x_data = tf.placeholder('float',[None,n_input])
y_data = tf.placeholder('float',[None,n_output])

weight = {
	'h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
	'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
	'out':tf.Variable(tf.random_normal([n_hidden_2,n_output]))
}
baise = {
	'h1':tf.Variable(tf.random_normal([n_hidden_1])),
	'h2':tf.Variable(tf.random_normal([n_hidden_2])),
	'out':tf.Variable(tf.random_normal([n_output]))
}

def neural_net(x):
	for key in weight:
		x = tf.add(tf.matmul(x,weight[key]),baise[key])
	return x

y = neural_net(x_data)
loss = tf.reduce_mean((y-y_data)**2)
optimizer = tf.train.AdamOptimizer(0.05)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for step in range(50000):
		batch_x,batch_y = mnist.train.next_batch(128)
		sess.run(train,feed_dict={x_data:batch_x,y_data:batch_y})
		if step %100 == 0:
			print(sess.run(loss,feed_dict={x_data:batch_x,y_data:batch_y}))
