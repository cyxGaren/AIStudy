
from numpy import *

def creatDataSet():
	group = array([[1,1],[0.9,1],[10000,0],[10000,0.1]])
	label = ['A','A','B','B']
	return group,label

def create2ndSet():
	group =[[1,1,1],
			[1,1,1],
			[1,0,0],
			[0,1,0],
			[0,1,0]]
	return group
