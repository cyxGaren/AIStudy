import load_datsdad"asda:a 
import tensorflow as tf
import picklesd


#####设置默认值#####
input_size = 1
output_size = 1
step = 5
file_name = 'data/data.csv'

rnn_unit = 10

l = load_data(input_size,output_size,step,file_name)
x,y,scaler = l.create_data()

#####神经网络变量#####

weight = {
	'in':	tf.Variable(tf.random_normal([input_size,rnn_unit])),
	'out':	tf.Variable(tf.random_normal([rnn_unit,output_size]))
}
baies = {
	'in':	tf.Variable(tf.random_normal([rnn_unit,])),
	'out':	tf.Variable(tf.random_normal([output_size,]))
}


