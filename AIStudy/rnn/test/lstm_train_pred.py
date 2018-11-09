import tensorflow as tf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy import *
from sklearn import preprocessing
import time
import sys
import pickle

step = 10       
batch_size = 30 

input_size = 1
output_size = 1
rnn_unit = 5




def load_data(filename,name):
        if (name=='rate'):
                data_rate = pd.read_csv(filename)['RATE']
        else:
                data_rate = pd.read_csv(filename)['THRESHOLD']
        data_rate=array(data_rate).reshape([-1,1])
        scaler = preprocessing.StandardScaler().fit(data_rate)
        data_rate = scaler.transform(data_rate)
        x,y = [],[]
        for i in range(len(data_rate)-step):
                x.append(data_rate[i:i+step])
                y.append(data_rate[i+1])
        return array(x).reshape([-1,step,input_size]),array(y).reshape([-1,output_size]),scaler



weight = {
        'in':   tf.Variable(tf.random_normal([input_size,rnn_unit])),
        'out':  tf.Variable(tf.random_normal([rnn_unit,output_size]))
}

baies = {
        'in':   tf.Variable(tf.random_normal([rnn_unit,])),
        'out':  tf.Variable(tf.random_normal([output_size,]))
}

X = tf.placeholder('float32',[None,step,input_size])
Y = tf.placeholder('float32',[None,output_size])

def lstm(batch):

        input_data = tf.reshape(X,[-1,input_size])
        rnn_data = tf.add(tf.matmul(input_data,weight['in']),baies['in'])
        rnn_data = tf.reshape(rnn_data,[-1,step,rnn_unit])

        cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
        h0 = cell.zero_state(batch,'float32')
        _,last_data = tf.nn.dynamic_rnn(cell,rnn_data,initial_state=h0)
        last_data = last_data.h
        
        pred_y = tf.add(tf.matmul(last_data,weight['out']),baies['out'])
        return pred_y

def train(name):
        global batch_size
        pred_y = lstm(batch_size)
        loss = tf.reduce_mean(tf.square(tf.reshape(pred_y,[-1])-tf.reshape(Y,[-1])))
        optimizer = tf.train.AdamOptimizer(0.006)
        train_op = optimizer.minimize(loss)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
                sess.run(init)
                num = 1000
                for i in range(num):
                        start = 0
                        end = start+batch_size
                        while end<len(train_y):
                                _loss,_ = sess.run([loss,train_op],feed_dict={X:train_x[start:end],Y:train_y[start:end]})
                                start =end
                                end = end+batch_size
                        if (i+1)%50 == 0:
                                print((i+1)/(num/100),'%')
                        saver.save(sess,'./ckpt/rnn_'+name+'.ckpt')




def use_model(name):
        pred_y = lstm(1)
        saver = tf.train.Saver()
        with tf.Session() as sess:
                saver.restore(sess,'./ckpt/rnn_'+name+'.ckpt')
                test_x_list_1 = train_x[-1:]
                test_x_list_2 = test_x
                pred_y_list_1 = []
                pred_y_list_2 = []
                for i in range(len(test_y)):
                        pred = sess.run(pred_y,feed_dict={X:test_x_list_1})
                        test_x_list_1[-1] = vstack((test_x_list_1[-1,1:],pred))
                        pred_y_list_1.extend(pred)
                        pred_y_list_2.extend(sess.run(pred_y,feed_dict={X:test_x_list_2[i:i+1]}))
                _test_y = scaler.inverse_transform(test_y)
                pred_y_list_1 = scaler.inverse_transform(pred_y_list_1)
                pred_y_list_2 = scaler.inverse_transform(pred_y_list_2)
                plt.subplot(211)
                plt.plot(range(0,index),scaler.inverse_transform(train_y),color='k')
                plt.plot(range(index,_length),_test_y,color='blue')
                plt.plot(range(index,_length),pred_y_list_2,color='green')
                plt.legend(['trainingData','testY','predY'])
                plt.subplot(234)
                plt.plot(range(index,_length),_test_y,color='blue')
                plt.subplot(236)
                plt.plot(range(index,_length),pred_y_list_2,color='green')
                plt.savefig('./pic/'+name+'.png')
                return pred_y_list_2


if(sys.argv[3]=='train'):
        time_start = time.time()
        filename,name = sys.argv[1],sys.argv[2]
        print('start training '+name+'.................')
        x,y,scaler = load_data(filename,name)
        index = (int)(3.0/7*len(x))
        _length = len(y)
        train_x,train_y,test_x,test_y = x[:index],y[:index],x[index:],y[index:]
        train(name)
        print('end training '+name+'.................')
        print('training '+name+' time: '+str(time.time()-time_start))

if(sys.argv[3]=='pred'):
        filename,name = sys.argv[1],sys.argv[2]
        x,y,scaler = load_data(filename,name)
        index = (int)(3.0/7*len(x))
        _length = len(y)
        train_x,train_y,test_x,test_y = x[:index],y[:index],x[index:],y[index:] 
        data = use_model(name)
        f=open('./ckpt/pred_'+name+'.txt','wb')
        pickle.dump(data,f)
        f.close()
