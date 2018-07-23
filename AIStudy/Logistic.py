from sklearn.linear_model import *
import matplotlib.pyplot as plt
import random

xs = range(100)
ys = []
for x in xs:
    y = (5*x+2+random.random()*50)
    ys.append(y)
plt.scatter(xs,ys)

print('xs',xs,'ys',ys)
arr = []
for i in range(100):
    tlist = []
    tlist.extend('1')
    tlist.extend('2')
    arr.append(tlist)
clf = LinearRegression()
clf.fit(arr,ys)
print(clf.coef_,clf.intercept_)
plt.show()