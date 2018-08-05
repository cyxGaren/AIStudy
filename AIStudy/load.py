import tensorflow as tf



xval = tf.Variable(324561,name='v1')


saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess,'ckpt/test.ckpt')
    print(sess.run(xval))