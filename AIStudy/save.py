import tensorflow as tf


x = tf.Variable(1,name='x')
W = tf.Variable([[1,1,1],[2,2,2]],dtype = tf.float32,name='w')
b = tf.Variable([[0,1,2]],dtype = tf.float32,name='b')
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    saver.save(sess,'ckpt/testsave.ckpt')