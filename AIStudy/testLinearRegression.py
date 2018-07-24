from numpy import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import *
from sklearn.pipeline import *
from sklearn.model_selection import train_test_split
from numpy import float64
from sklearn import *
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import ExtraTreeRegressor

score_init = 0
model_init = ''
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

def load_data():
    data = pd.read_csv('test.csv')
    data.dropna(inplace=True)
    X = data[["R001_014","R001_016","R001_018","R001_020","R001_022","R001_013","R001_015","R001_017","R001_019","R001_021"]]
    Y = data ['RATE']
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)
    return X_train,X_test,Y_train,Y_test


def try_different_method(model):
    x_train, x_test, y_train, y_test = load_data()
    model.fit(x_train,y_train)
    score = model.score(x_test, y_test)
    find_max_score(score,model)


def polt_max_score(model):
    x_train, x_test, y_train, y_test = load_data()
    model.fit(x_train,y_train)
    result = model.predict(x_test)
    # print_y = model.predict(x_test[0:50])
    # array_y = array(y_test).reshape(-1,1)
    # out = []
    # for i in range(50):
    #     lists = []
    #     lists.append(print_y[i])
    #     lists.append(array_y[i])
    #     out.append(lists)
    # print(out)
    plt.figure()
    plt.plot(arange(len(result)), y_test,'go-',label='true value')
    plt.plot(arange(len(result)),result,'ro-',label='predict value')
    plt.title('score: %f  model:%s'%(score_init,model.__class__.__name__))
    plt.legend()
    plt.show()

def find_max_score(score,model):
    global score_init
    global model_init
    if score_init < score:
        model_init = model.__class__.__name__
        score_init = score

def create_model():
    dict = {}
    ####3.1决策树回归####
    model_DecisionTreeRegressor = tree.DecisionTreeRegressor()
    ####3.2线性回归####
    model_LinearRegression = linear_model.LinearRegression()
    ####3.3SVM回归####
    model_SVR = svm.SVR()
    ####3.4KNN回归####
    model_KNeighborsRegressor = neighbors.KNeighborsRegressor()
    ####3.5随机森林回归####
    model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)#这里使用20个决策树
    ####3.6Adaboost回归####
    model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=50)#这里使用50个决策树
    ####3.7GBRT回归####
    model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)#这里使用100个决策树
    ####3.8Bagging回归####
    model_BaggingRegressor = BaggingRegressor()
    ####3.9ExtraTree极端随机树回归####
    model_ExtraTreeRegressor = ExtraTreeRegressor()

    dict['DecisionTreeRegressor'] = model_DecisionTreeRegressor
    dict['LinearRegression'] = model_LinearRegression
    dict['SVR'] = model_SVR
    dict['KNeighborsRegressor'] = model_KNeighborsRegressor
    dict['RandomForestRegressor'] = model_RandomForestRegressor
    dict['AdaBoostRegressor'] = model_AdaBoostRegressor
    dict['GradientBoostingRegressor'] = model_GradientBoostingRegressor
    dict['BaggingRegressor'] = model_BaggingRegressor
    dict['ExtraTreeRegressor'] = model_ExtraTreeRegressor

    return dict

def use_model():
    global model_init
    dict = {}
    dict_model = create_model()
    for i in range(10):
        for key in dict_model:
            try_different_method(dict_model[key])
        if model_init not in dict.keys():
            dict[model_init] = {}
        dict[model_init]['count'] = dict[model_init].get('count',0)+1
        dict[model_init]['score'] = (dict[model_init].get('score',0)\
                                    *(dict[model_init]['count']-1)+score_init)\
                                    /dict[model_init]['count']

    sort_list = [[dict[example]['count'],example] for example in dict]
    sort_list = array(sort_list)
    index = array(sort_list[:,0]).argsort()[-1]
    model = sort_list[index][1]
    polt_max_score(dict_model[model])
use_model()

