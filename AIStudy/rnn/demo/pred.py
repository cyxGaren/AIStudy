import load_data
import tensorflow as tf
import pickle



#####设置默认值#####
input_size = 1
output_size = 1
step = 5
file_name = 'data/data.csv'

rnn_unit = 10
batch_size = 1	#每次40组
lr = 0.006
train_count = 1000

sess_save_file = 'data/ckpt/sess.ckpt'


################################定义相同的网络结构#############################
#####神经网络变量#####

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





class Pred:
#####预测#####
#####只预测一个#####
    # def __init__(self):

    def get_pred(self,x):
        self.x = x
        pred_y = self.lstm(batch_size)
        saver = tf.train.Saver()
        #####迭代训练#####
        with tf.Session() as sess:
            saver.restore(sess, sess_save_file)
            return sess.run(pred_y, feed_dict={X: self.x})


    #####神经网络#####
    def lstm(self,batch):
        #三维转二维
        _X = tf.reshape(X,[-1,input_size])
        #二维转三维
        input_data = tf.reshape(tf.add(tf.matmul(_X,weight['in']),baies['in']),[-1,step,rnn_unit])
        #定义神经元
        cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
        h0 = cell.zero_state(batch,'float32')
        _,last_data = tf.nn.dynamic_rnn(cell,input_data,initial_state=h0)
        last_data = last_data.h
        return tf.add(tf.matmul(last_data,weight['out']),baies['out'])
    ################################定义相同的网络结构#############################

