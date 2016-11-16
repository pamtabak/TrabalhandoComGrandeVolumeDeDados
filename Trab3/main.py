#import matplotlib as plt
import numpy as np
import csv
import math
import scipy.stats as stats
import pandas as pd

# Target is the real class of the object (last attribute)
# Target 1 is the less likely (and Target 0 the most likely)

header = []

def readDataset (datasetName, getHeaders = False):
    # Array of arrays. Each array is a column of the dataset
    dataset = []

    # Reading dataset
    csvFile   = open(datasetName, 'rb')
    csvReader = csv.reader(csvFile, delimiter=',')

    # Reading headers (so we know the amount of columns there are)
    row = csvReader.next()
    for column in row:
        dataset.append([])
        if (getHeaders):
                header.append(column)

    for row in csvReader:
        col = 0
        for column in row:
            dataset[col].append(float(column))
            col += 1

    return dataset

trainingDataset = readDataset('dataset_trabalho3/train_file.csv', True)
testingDataset  = readDataset('dataset_trabalho3/test_file.csv')

# Checking what we can do with each column
columnsToRemove = []
indexesToRemove = []
headersToRemove = []

# 1. If there are any 2 columns correlated
eps = 0.05
for index1 in xrange(len(trainingDataset)):
    if (index1 not in indexesToRemove):
        for index2 in xrange(index1, len(trainingDataset)):
            if (index1 != index2):
                if (index2 not in indexesToRemove):
                    correlation = stats.pearsonr(trainingDataset[index1], trainingDataset[index2])[0]
                    if (abs(correlation) + eps >= 1):
                        indexesToRemove.append(index2)
                        columnsToRemove.append(trainingDataset[index2])
                        headersToRemove.append(header[index2])
    print(index1)

for column in columnsToRemove:
    index = trainingDataset.index(column)
    trainingDataset.remove(column)
    del testingDataset[index]

for h in headersToRemove:
    header.remove(h)

#2. If there are any columns that are "indexes"
hashingTrickColumns = []
# Hashing Trick (look for columns that have less than 50 unique values on it)
uniqueValues = 2
for column in trainingDataset:
        # We need to be sure we won`t remove the TARGET column
        if (len(set(column)) < uniqueValues and trainingDataset.index(column) != len(trainingDataset) - 1):
            hashingTrickColumns.append(column)

# transposeTrainingDataset = np.transpose(trainingDataset).tolist()
# trainingDf               = pd.DataFrame(data=transposeTrainingDataset, columns=header)
# testingDf  = pd.DataFrame(data=testingDataset,  columns=header[:-1])


# salvar dataset sem correlacao em um novo csv (salvar tambem o de teste)
np.savetxt('newTrainingDataset.csv', np.transpose(trainingDataset), delimiter=",")
np.savetxt('newTestingDataset.csv',  np.transpose(testingDataset),  delimiter=",")

# rodar processo de classificacao nos dois datasets com um cross-validation com k pelo menos 50
# ver se melhorou
