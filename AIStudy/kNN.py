
from createNum import *
import operator

oldgroup,labels = creatDataSet()

def getLabel(val,k):
	group = autoSet(oldgroup)
	height = group.shape[0]
	valArr = tile(val,(height,1))
	newArr = valArr - group
	newArr **= 2
	distance = newArr.sum(axis=1)
	index = argsort(distance)
	result = {}
	for i in range(k):
		if i == len(index)-1:
			break
		label = labels[index[i]]
		result[label] = result.get(label,0)+1
	sortedResult = sorted(result.items(),key=lambda val:val[1],reverse=True)

	print(sortedResult[0][0])
	

def autoSet(val):
	mins = val.min(0)
	maxs = val.max(0)
	ranges = maxs - mins
	dataSet = zeros(val.shape)
	dataSet = val - tile(mins,(val.shape[0],1))
	dataSet /= tile(ranges,(val.shape[0],1))
	return dataSet
	

getLabel([123,0],3)

