from numpy import *
import pandas as pd
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt


step = 10   #设为前面十个影响后面
batch_size = 30 #每次30个样本
input_size = 1      #batch*step*input_size
output_size = 1
rnn_unit = 5    #隐层大小
lr = 0.006

def create_data():

    x = array(range(400))
    y = sin(x)+sqrt(x)/100+random.rand(400)*sqrt(x)
    y = y.reshape([-1,input_size])  #此处需要将y转为2维,后续train_y转3维
    train_x,train_y = [],[]
    for i in range(len(x)-step):
        train_x.append(y[i:(i+step)].tolist())
        train_y.append(y[i+step].tolist())
    return array(train_x),array(train_y)
    #return train_x,train_y


#define variable
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

#define RNN
def lstm(batch_size):
    w_in = weight['in']
    w_out = weight['out']
    b_in = baise['in']
    b_out = baise['out']
    #3维转2维
    input_data = tf.reshape(X,[-1,input_size])
    rnn_data = tf.add(tf.matmul(input_data,w_in),b_in)
    #2维转3维,做cell输入
    rnn_data = tf.reshape(rnn_data,[-1,step,rnn_unit])
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    h0 = cell.zero_state(batch_size,'float32')
    rnn_output,rnn_last = tf.nn.dynamic_rnn(cell,rnn_data,initial_state=h0)
    rnn_last = tf.reshape(rnn_last.h,[-1,rnn_unit])
    pred_y = tf.add(tf.matmul(rnn_last,w_out),b_out)
    return pred_y


train_x,train_y = create_data()

def train():
    global batch_size
    pred_y = lstm(batch_size)
    loss = tf.reduce_mean(tf.square(tf.reshape(pred_y,[-1])-tf.reshape(Y,[-1])))
    optimizer = tf.train.AdamOptimizer(lr)
    train_op = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        min_loss = 9999
        for i in range(10):
            start = 0
            j=0
            end = start+batch_size
            while end<len(train_x):
                _loss,_optimizer = sess.run([loss,train_op],feed_dict={X:train_x[start:end],Y:train_y[start:end]})
                start += batch_size
                end += batch_size
                if j%10 ==0:
                    min_loss = min(_loss,min_loss)
                j+=1
            if (i+1) % 100 ==0:
                saver.save(sess, 'ckpt/rnn.ckpt')
            print(min_loss)
        plt.plot(train_y)
        start = 0
        end = start + batch_size
        yy = []
        while end < len(train_x):
            yy.extend(sess.run([pred_y], feed_dict={X: train_x[start:end]}))
            start += batch_size
            end += batch_size
        yy = array(yy).reshape([-1])
        print(train_y.shape,array(yy).shape)
        plt.plot(yy)
        plt.legend(['train','yy'])
        plt.show()


def load():
    pred_y = lstm(1)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,'ckpt/rnn.ckpt')
        for i in range(100):
            yy = sess.run(pred_y,feed_dict={X:train_x[i:i+1]})
            print(yy,train_y[i])

# load()

train()