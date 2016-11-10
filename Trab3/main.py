import matplotlib as plt
import numpy as np
import csv
import math
import scipy.stats as stats

# Target is the real class of the object (last attribute)
# Target 1 is the less likely (and Target 0 the most likely)

header = []

def readDataset (datasetName, getHeaders = false):
	# Array of arrays. Each array is a column of the dataset
	dataset = []

	# Reading dataset
	csvFile   = open(datasetName, 'rb')
	csvReader = csv.reader(csvFile, delimiter=',')

	# Reading headers (so we know the amount of columns there are)
	row = csvReader.next()
	for column in row:
		dataset.append([])
        #if (getHeaders):
                #header.append(column)

	for row in csvReader:
		col = 0
		for column in row:
			dataset[col].append(float(column))
			col += 1

	return dataset

trainingDataset = readDataset('dataset_trabalho3/train_file.csv', true)
testingDataset  = readDataset('dataset_trabalho3/test_file.csv')

print(header)
# Checking what we can do with each column


# 1. If there is any column that doesn`t make any difference (for instance, all the records are filled with the same value)
#2. If there are any columns that are "indexes"
columnsToRemove = []
for column in trainingDataset:
        size = len(set(column))
        print(size)
        if (len(set(column)) < 2 and trainingDataset.index(column) != len(trainingDataset) -1 ):
                columnsToRemove.append(column)

for column in columnsToRemove:
        index = trainingDataset.index(column)
        trainingDataset.remove(column)
        del testingDataset[index]

columnsToRemove = []
print(len(trainingDataset))


# 3. If there are any 2 columns correlated
# eps   = 0.05
# for column1 in trainingDataset:
# 	sameColumns = False
# 	for column2 in trainingDataset:
# 		# Just checking if the columns are not the same
# 		if (np.array_equal(column1, column2) == False):
# 			correlation = stats.pearsonr(column1, column2)[0]
# 			if (math.fabs(correlation) + eps >= 1):
# 				print(len(trainingDataset))
# 				trainingDataset.remove(column2)
#                 testingDataset.remove(column2)
# salvar dataset sem correlacao em um novo csv (salvar tambem o de teste)
# npTrainingDataset = np.array(trainingDataset)
# npTestingDataset = np.array(testingDataset)
# np.savetxt('trainingDataset.csv', npTrainingDataset, delimiter=",")
# np.savetxt('testingDataset.csv', npTestingDataset, delimiter=",")

# rodar processo de classificacao nos dois datasets com um cross-validation com k pelo menos 50
# ver se melhorou
