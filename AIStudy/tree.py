from createNum import *
from math import log


def calcShano(group):
	numEntries = len(group)
	label = {}
	for list in group:
		label[list[-1]] = label.get(list[-1],0)+1
	shanoEnt = 0
	for key in label.keys():
		percent = label[key]/numEntries
		shanoEnt -= percent*log(percent,2)
	return shanoEnt

def subList(group,index,value):
	result = []
	for list in group:
		if list[index]==value:
			newList = list[:index]
			newList.extend(list[index+1:])
			result.append(newList)
	return result

def choseBestFeatureToSplit():
	bestFeature = -1
	bestShano = 1
	
	group = create2ndSet()
	length = len(group[0])-1
	for i in range(length):
		list = [example[i] for example in group]
		listSet = set(list)
		newShano = 0
		for val in listSet:
			splitList = subList(group,i,val)
			prod = len(splitList)/len(group)
			newShano += calcShano(splitList)*prod
		if newShano <= bestShano:
			bestShano = newShano
			bestFeature = i
	return bestFeature

print(choseBestFeatureToSplit())
