from numpy import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf

step = 5
lr = 0.003

batch_size = 395
input_size = 1  #   batch_size*step*input_size
output_size = 1
rnn_unit = 10   #隐层大小

def create_data():
    data = (arange(400))
    data = sin(data)+data/100+random.rand(400)*sqrt(data)
    data = [float32(example) for example in data]
    x = []
    y = []
    for i in range(len(data)-step):
        x.append(data[i:(i+step)])
        y.append(data[i+step])
    return x,y

x,y = create_data()
x = array(x).reshape([-1,5,1])
y = array(y).reshape([-1,1])

weight = {
    'in':   tf.Variable(tf.random_normal([input_size,rnn_unit])),
    'out':  tf.Variable(tf.random_normal([rnn_unit,output_size]))
}

biase = {
    'in':   tf.Variable(tf.random_normal([rnn_unit,])),
    'out':  tf.Variable(tf.random_normal([1,]))
}

def lstm(batch_size):
    w_in = weight['in']
    w_out = weight['out']
    b_in = biase['in']
    b_out = biase['out']
    x_data = tf.reshape(x,[-1,input_size])
    rnn_data = tf.matmul(x_data,w_in)+b_in
    rnn_data = tf.reshape(rnn_data,[-1,step,rnn_unit])
    #调用lstm
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_val = cell.zero_state(batch_size,'float32')
    rnn_output,rnn_last = tf.nn.dynamic_rnn(cell,rnn_data,initial_state=init_val)
    print(rnn_last)
    y_data = tf.matmul(rnn_last.h,w_out)+b_out
    return y_data


pred_y = lstm(batch_size)
loss = tf.reduce_mean(tf.square(tf.reshape(pred_y,[-1])-tf.reshape(y,[-1])))
optimizer = tf.train.AdamOptimizer(lr)
train = optimizer.minimize(loss)

saver = tf.train.Saver()

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(10000):
        the_train,the_loss = sess.run([train,loss])
        if (i+1)%50 == 0:
            print(the_loss)
    yy = sess.run(pred_y)
    #saver.save(sess,'ckpt/test.ckpt')
    plt.plot(y)
    plt.plot(yy)
    plt.show()
# saver2 = tf.train.Saver()
# path = tf.train.latest_checkpoint('ckpt/')
# print(path)
# with tf.Session() as sess:
#     saver2.restore(sess,path)