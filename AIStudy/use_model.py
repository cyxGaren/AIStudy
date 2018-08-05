import tensorflow as tf


xval = tf.Variable(123,name='v1')

# path = tf.train.latest_checkpoint('ckpt/')
# print(path)
# init = tf.global_variables_initializer()
saver = tf.train.Saver()
# graph = tf.Graph()
#
# saver = tf.train.import_meta_graph('ckpt/test.ckpt-1.meta')
with tf.Session() as sess:
    saver.restore(sess,'ckpt/test.ckpt-1')
    print(sess.run(xval))