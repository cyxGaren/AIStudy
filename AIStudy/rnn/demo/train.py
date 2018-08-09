import load_data
import tensorflow as tf
import pickle


#####设置默认值#####
input_size = 1
output_size = 1
step = 5
file_name = 'data/data.csv'

rnn_unit = 10
batch_size = 40	#每次40组
lr = 0.006
train_count = 1000

sess_save_file = 'data/ckpt/sess.ckpt'


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

#####神经网络#####
def lstm(batch):
    #三维转二维
    X = tf.reshape(X,[-1,input_size])
    #二维转三维
    input_data = tf.reshape(tf.add(tf.matmul(X,weight['in'])),[-1,step,rnn_unit])
    #定义神经元
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    h0 = cell.zero_state(batch,'float32')
    _,last_data = tf.nn.dynamic_rnn(cell,input_data,initial_state=h0)
    last_data = last_data.h
    return tf.add(tf.matmul(last_data,weight['out']),baies['out'])

#####训练#####
def train():
    pred_y = lstm(batch_size)
    loss = tf.reduce(tf.square(tf.reshape(pred_y,[-1])-tf.reshape(Y,[-1])))
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    #####迭代训练#####
    with tf.Session() as sess:
        sess.run(init)
        for i in range(train_count):
            #####batch流#####
            strat = 0
            end = start+batch_size
            while end<x_len:
                sess.run(optimizer,feed_dict={X:x[start:end],Y:y[satrt:end]})
                start = end
                end += batch_size
            if i==0 or (i+1)%50==0:
                print((i+1)/x_len*100+'%')
                saver.save(sess,sess_save_file)


l = load_data(input_size,output_size,step,file_name)
x,y,scaler = l.create_data()

x_len = len(x)

train()