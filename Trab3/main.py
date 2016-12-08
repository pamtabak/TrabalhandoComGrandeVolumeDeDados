#import matplotlib as plt
import numpy as np
import csv
import math
import scipy.stats as stats
import pandas as pd

# Target is the real class of the object (last attribute)
# Target 1 is the less likely (and Target 0 the most likely)

header = []

def readDataset (datasetName, numberOfColumns, removeHeader = False):
    # Array of arrays. Each array is a column of the dataset
    dataset = []

    # Reading dataset
    csvFile   = open(datasetName, 'rb')
    csvReader = csv.reader(csvFile, delimiter=',')

    if (removeHeader):
        row = csvReader.next()
        if (len(header) == 0):
            for column in row:
                header.append(column)

    for i in xrange(0,numberOfColumns):
        dataset.append([])

    for row in csvReader:
        col = 0
        for column in row:
            dataset[col].append(float(column))
            col += 1

    return dataset

def removeCorrelation (eps = 0.05):
    # Checking what we can do with each column
    columnsToRemove = []
    indexesToRemove = []
    headersToRemove = []

    # 1. If there are any 2 columns correlated
    for index1 in xrange(len(testingDataset)):
        if (index1 not in indexesToRemove):
            for index2 in xrange(index1, len(testingDataset)):
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

def createColumn():
    newHeader = []
    for col in xrange(0, len(trainingDataset)):
        newHeader.append("A" + str(col));
    return newHeader;

def hashingTrick(newHeader, uniqueValues = 50):
    print("dentro da funcao hashing trick")
    #If there are any columns that are "indexes"
    hashingTrickColumns = [] #name of the column that should have a hashing trick operation
    # Hashing Trick (look for columns that have less than 50 unique values on it)
    for column in xrange(0, len(trainingDataset) - 1):
        # We need to be sure we won`t remove the TARGET column
        if (len(set(trainingDataset[column])) < uniqueValues):
            hashingTrickColumns.append(newHeader[column])

    print(hashingTrickColumns)

    trainingLines = len(trainingDataset[0])
    testingLines  = len(testingDataset[0])

    mergedDataset = np.concatenate((trainingDataset[:-1], testingDataset), axis=1)

    df = pd.DataFrame(data=np.transpose(mergedDataset), columns=newHeader[:-1])

    for column in hashingTrickColumns:
        just_dummies = pd.get_dummies(df[column])
        df = pd.concat([df, just_dummies], axis=1)
        df.drop([column], inplace=True, axis=1)

    # Cant concat because the second parameter is not a pandas dataframe
    # trainingDf = pd.concat([df[0:trainingLines], trainingDataset[-1]], axis = 1)
    trainingDf = pd.concat([df[0:trainingLines],pd.DataFrame(trainingDataset[-1], columns=["target"])], axis=1)
    testingDf  = df[trainingLines:]

    print(len(trainingDf.columns))
    print(len(testingDf.columns))

    trainingDf.to_csv('resultados/trainingWithHashTrick.csv', index = False, header= False)
    testingDf.to_csv('resultados/testingWithHashTrick.csv', index = False, header= False)


#############################################################################
trainingDataset = readDataset('dataset_trabalho3/train_file.csv', 370, removeHeader=True)
testingDataset  = readDataset('dataset_trabalho3/test_file.csv', 369, removeHeader=True)

print(header)

removeCorrelation()
newTrainingDataset = np.insert(trainingDataset,0,header, axis=1)
newTestingDataset  = np.insert(testingDataset,0,header[-1:], axis=1)

# newTrainingDataset = header + trainingDataset
# newTestingDataset  = header[:-1] + testingDataset
print(len(newTrainingDataset))
print(len(newTestingDataset))
np.savetxt('dataset_trabalho3/train_without_correlation.csv', newTrainingDataset, fmt="%f", delimiter=",")
np.savetxt('dataset_trabalho3/test_without_correlation.csv', newTestingDataset, fmt="%f", delimiter=",")

# newHeader = createColumn()
# hashingTrick(newHeader)

