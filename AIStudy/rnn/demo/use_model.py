import tensorflow as tf
import load_data
import pred

l = load_data.Load_data()
x, y = l.create_data()
scaler = l.get_scaler()

class T:
    def __init__(self,scaler,x,y):
        _scaler = scaler['x']
        p = pred.Pred()
        print(_scaler.inverse_transform(p.get_pred(x[0:1])),_scaler.inverse_transform(y[0:1]))

t = T(scaler,x,y)
#with tf.variable_scope("a"):
#	t = T(scaler,x,y)
