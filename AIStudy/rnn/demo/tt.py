import tensorflow as tf

class fun:
	def __init__(self):
		X = tf.Variable(25)

		init = tf.global_variables_initializer()

		saver = tf.train.Saver()
		'''
		with tf.Session() as sess:
			sess.run(init)
			saver.save(sess,'t/ttt.ckpt')
			print('ok')
		'''
		with tf.Session() as sess:
			saver.restore(sess,'t/ttt.ckpt')
			print(sess.run(X))

for i in range(5):
	fun()
