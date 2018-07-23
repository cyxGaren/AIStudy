from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from numpy import *
import random

xs = array([6,2,8,1,1,10,2,14,3,18]).reshape(-1,2)
ys = []
for x in xs: 
	y = (x[0]*2+x[1]+random.random())
	ys.append(y)
ys = array(ys).reshape(-1,1)


func = LinearRegression()
func.fit(xs,ys)
print(func.intercept_,func.coef_)
print("输出:尺寸为12的价格:$%.2f"%func.predict(matrix([8,1])))

ridge = RidgeCV(alphas = [0.01,0.05,0.1,0.2,0.5,1,5])
ridge.fit(xs,ys)
print(ridge.intercept_,ridge.coef_)
print("输出:尺寸为12的价格:$%.2f"%ridge.predict(matrix([8,1])))



fr = open('ex0.txt')
dataMat = []
labelMat = []
for line in fr.readlines():
	lineArr = []
	curLine = line.strip().split('\t')
	for i in range(2):
		lineArr.append(curLine[i])
	dataMat.append(lineArr)
	labelMat.append(curLine[-1])
ridge = RidgeCV(alphas = [0.01,0.05,0.1,0.2,0.5,1,5])
ridge.fit(dataMat,labelMat)

print(ridge.coef_)

