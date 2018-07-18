from numpy import *

def createTrainMatrix():
	trainMatrix = [['my','dog','has','flea','problems','help','plz'],
		['maybe','not','take','him','to','dog','park','stupid'],
		['my','pet','is','so','cute','I','love','him'],
		['stop','stupid','garbage'],
		['mr','lickks','ate','my','steak','how','to','stop','him'],
		['quit','buying','worth','less','dog','food','stupid']]
	classVec = [0,1,0,1,0,1]	
	return trainMatrix,classVec

def createVocabList(trainMatrix):
	vocabSet = set([])
	for trainList in trainMatrix:
		vocabSet = vocabSet | set(trainList)
	vocabList = list(vocabSet)
	vocabList.sort()
	return vocabList

def bagOfVocab(inputMatrix,vocabList):
	outputMatrix = []
	for inputList in inputMatrix:
		outputList = [0]*len(vocabList)
		for word in inputList:
			outputList[vocabList.index(word)] += 1
		outputMatrix.append(outputList)
	return outputMatrix

def trainNB():
	trainMatrix,classVec = createTrainMatrix()
	vocabList = createVocabList(trainMatrix)
	pNormal = sum(classVec)/float(len(classVec))
	outputMatrix = bagOfVocab(trainMatrix,vocabList)
	numList = len(outputMatrix)
	numWord = len(outputMatrix[0])

	vecBad = ones(numWord)
	vecNormal = ones(numWord)
	countBad = countNormal = 0.0
	for i in range(numList):
		if classVec[i]==1:
			vecBad += outputMatrix[i]
			countBad += sum(outputMatrix[i])
		else:
			vecNormal += outputMatrix[i]
			countNormal += sum(outputMatrix[i])
	wordBadP = log(vecBad/float(countBad))
	wordNormalP = log(vecNormal/float(countNormal))
	return pNormal,wordBadP,wordNormalP

def testingNB():
	trainMatrix,classVec = createTrainMatrix()
	vocabList = createVocabList(trainMatrix)
	test1 = [['stupid','my','dog']]
	test2 = [['I','love','dog']]
	vec1 = bagOfVocab(test1,vocabList)
	vec2 = bagOfVocab(test2,vocabList)
	pNormal,wordBadP,wordNormalP = trainNB()
	print(classifyNB(pNormal,wordBadP,wordNormalP,vec1))
	print(classifyNB(pNormal,wordBadP,wordNormalP,vec2))
	

def classifyNB(pNormal,wordBadP,wordNormalP,vec1):
	pB = sum(wordBadP * vec1) + log(1-pNormal)
	pN = sum(wordNormalP * vec1) + log(pNormal)
	if pB > pN:
		return 'bad'
	else:
		return 'good'

testingNB()
