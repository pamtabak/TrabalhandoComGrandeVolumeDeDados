from sklearn.ensemble import RandomForestClassifier
import numpy as np
import csv

#3961346

def readDataset (datasetName, numberOfColumns):
    # Array of arrays. Each array is a column of the dataset
    dataset = []

    # Reading dataset
    csvFile   = open(datasetName, 'rb')
    csvReader = csv.reader(csvFile, delimiter=',')

    for i in xrange(0,numberOfColumns):
        dataset.append([])

    for row in csvReader:
        col = 0
        for column in row:
            dataset[col].append(float(column))
            col += 1

    return dataset

trainingDataset = readDataset('dataset_trabalho3/newTrainingDataset.csv', 240)
testingDataset  = readDataset('dataset_trabalho3/newTestingDataset.csv', 239)

target = trainingDataset[-1:][0]
train  = trainingDataset[0:len(trainingDataset)-1]
test   = testingDataset

#create and train the random forest
#multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(np.transpose(train), target)

result = np.transpose(rf.predict_proba(np.transpose(test)))[1]
# print(len(result))
# print(len(np.transpose(result)))

np.savetxt('result.csv', result, fmt="%f", delimiter=",")