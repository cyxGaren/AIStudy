from numpy import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import *
from sklearn.pipeline import *
from sklearn.model_selection import train_test_split
from numpy import float64
def test1():
    data = pd.read_csv('test.csv')
    data.dropna(inplace=True)
    X = data[["R001_014","R001_016","R001_018","R001_020","R001_022","R001_013","R001_015","R001_017","R001_019","R001_021"]]
    Y = data ['RATE']
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)

    func = LinearRegression()
    func.fit(X_train,Y_train)
    Y_pred = func.predict(X_test)

    ridge = Ridge()
    ridge.fit(X_train,Y_train)
    Y_pred_ridge = ridge.predict(X_test)

    sum_mean = 0

    Y_test = array(Y_test)

    for i in range(len(Y_pred)):
        sum_mean += sqrt((Y_pred[i] - Y_test[i])**2)

    print(sum_mean)

    plt.figure()
    plt.plot(range(len(Y_pred)), Y_pred, 'b', label="predict")
    plt.plot(range(len(Y_pred)), Y_test, 'r', label="test")
    plt.plot(range(len(Y_pred)), Y_pred_ridge, 'y', label="ridge")

    plt.legend(loc="upper right")
    plt.show()

test1()
#
# sns.pairplot(data,x_vars=["R001_014","R001_016","R001_018","R001_020","R001_022",
#                           "R001_013","R001_015","R001_017","R001_019","R001_021"],
#              y_vars=['RATE'],kind='reg')
#
#
# plt.show()
