import tensorflow as tf
import numpy as np


x_data = (np.linspace(-1,1,100)).reshape(-1,1)
noice = np.random.normal(0,0.5,x_data.shape)
y_data = np.square(x_data)+noice
print(y_data)
