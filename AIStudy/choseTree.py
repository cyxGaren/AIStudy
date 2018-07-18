from numpy import *
from math import *

group = [[1,0,0,'w'],[0,0,1,'d'],[0,1,0,'m'],[0,1,0,'m'],[1,1,0,'w']]
labels = ['arrom','magic','knife']

def createTree(group,labels):
	resultList = [example[-1] for example in group]
	resultSet = set(resultList)
	if len(resultSet)==1:
		return resultSet.pop()
	if len(resultList)==1:
		return mostValue
	bestFeature = choseBestFeature(group,labels)
	bestFeatLabel = labels[bestFeature]
	myTree = {bestFeatLabel:{}}
	featSet = set([example[bestFeature] for example in group])
	for value in featSet:
		subLabels = labels[:bestFeature]
		subLabels.extend(labels[bestFeature+1:])
		myTree[bestFeatLabel][value] = createTree(splitList(group,bestFeature,value),subLabels)
	return myTree	





def choseBestFeature(group,labels):
	bestShanoEnt = calcShano(group)
	bestFeature = -1
	for i in range(len(labels)):
		valueSet = set([example[i] for example in group])
		newShanoEnt = 0.0
		for value in valueSet:
			splitGroup = splitList(group,i,value)
			prod = len(splitGroup)/float(len(group))
			newShanoEnt += prod*calcShano(splitGroup)
		if newShanoEnt<bestShanoEnt:
			bestShanoEnt = newShanoEnt
			bestFeature = i
	return bestFeature

def splitList(group,index,value):
	splitGroup = []
	for list in group:
		if list[index]==value:
			appendList = list[:index]
			appendList.extend(list[index+1:])
			splitGroup.append(appendList)
	return splitGroup

def calcShano(group):
	groupLen = len(group)
	shanoEnt = 0.0
	dict = {}
	for list in group:
		resultLabel = list[-1]
		dict[resultLabel] = dict.get(resultLabel,0)+1
	for key in dict.keys():
		prod = dict[key]/groupLen
		shanoEnt -= prod*log(prod,2)
	return shanoEnt

print(createTree(group,labels))	
	
