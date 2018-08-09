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
        a = _scaler.inverse_transform(p.get_pred(x[0:1])),_scaler.inverse_transform(y[0:1])
