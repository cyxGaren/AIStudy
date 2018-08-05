from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.externals.six import StringIO
import pydotplus as pydot
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
# dot_data = StringIO()
# iris = load_iris()
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(iris.data, iris.target)
# tree.export_graphviz(clf, out_file=dot_data)
# graph = pydot.graph_from_dot_data(dot_data.getvalue())
# graph.write_pdf("iris.pdf")

def load_data():
    data = pd.read_csv('test.csv')
    data.dropna(inplace=True)
    X = data[["R001_014","R001_016","R001_018","R001_020","R001_022","R001_013","R001_015","R001_017","R001_019","R001_021"]]
    Y = data ['RATE']
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)
    clf = RandomForestRegressor()
    clf.fit(X_train,Y_train)
    dot_data = StringIO()
    tree.export_graphviz(clf,out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("No2.pdf")

load_data()


