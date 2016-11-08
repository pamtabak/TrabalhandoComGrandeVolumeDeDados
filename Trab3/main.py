import matplotlib as plt
import numpy as np
import csv
import math
import scipy.stats as stats

# Target is the real class of the object (last attribute)
# Target 1 is the less likely (and Target 0 the most likely)

def readDataset (datasetName):
	# Array of arrays. Each array is a column of the dataset
	dataset = []

	# Reading dataset
	csvFile   = open(datasetName, 'rb')
	csvReader = csv.reader(csvFile, delimiter=',')

	# Reading headers (so we know the amount of columns there are)
	row = csvReader.next()
	for column in row:
		dataset.append([])

	for row in csvReader:
		col = 0
		for column in row:
			dataset[col].append(float(column))
			col += 1

	return dataset


trainingDataset = readDataset('dataset_trabalho3/train_file.csv')
testingDataset  = readDataset('dataset_trabalho3/test_file.csv')

# Checking what we can do with each column
# 1. If there is any column that doesn`t make any difference (for instance, all the records are filled with the same value)
for column in trainingDataset:
	allElementsEqual = True
	value = column[0]
	for x in column:
		if (x != value):
			allElementsEqual = False
			break
	if (allElementsEqual):
		trainingDataset.remove(column)
        testingDataset.remove(column)


#2. If there are any 2 columns correlated
eps   = 0.05
for column1 in trainingDataset:
	sameColumns = False
	for column2 in trainingDataset:
		# Just checking if the columns are not the same
		if (np.array_equal(column1, column2) == False):
			correlation = stats.pearsonr(column1, column2)[0]
			if (math.fabs(correlation) + eps >= 1):
				print(len(trainingDataset))
				trainingDataset.remove(column2)
                testingDataset.remove(column2)
# salvar dataset sem correlacao em um novo csv (salvar tambem o de teste)
# rodar processo de classificacao nos dois datasets com um cross-validation com k pelo menos 50
# ver se melhorou
