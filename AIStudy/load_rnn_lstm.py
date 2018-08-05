from numpy import *
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

step = 10   #设为前面十个影响后面
batch_size = 30 #每次30个样本
input_size = 1      #batch*step*input_size
output_size = 1
rnn_unit = 5    #隐层大小
lr = 0.006

def load():
    # path = tf.train.latest_checkpoint('ckpt/')
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    x,y = create_data()
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess,'ckpt/rnn.ckpt')
        yy = sess.run(pred,feed_dict={X:x[:30]})

def create_data():
    x = array(range(400))
    y = sin(x)+sqrt(x)/100+random.rand(400)*sqrt(x)
    y = y.reshape([-1,input_size])  #此处需要将y转为2维,后续train_y转3维
    train_x,train_y = [],[]
    for i in range(len(x)-step):
        train_x.append(y[i:(i+step)].tolist())
        train_y.append(y[i+step].tolist())
    return array(train_x),array(train_y)


weight = {
    'in':   tf.Variable(tf.random_normal([input_size,rnn_unit])),
    'out':  tf.Variable(tf.random_normal([rnn_unit,output_size]))
}
baise = {
    'in':   tf.Variable(tf.random_normal([rnn_unit,])),
    'out':  tf.Variable(tf.random_normal([output_size,]))
}
X = tf.placeholder('float',[None,step,input_size])
Y = tf.placeholder('float',[None,output_size])


load()