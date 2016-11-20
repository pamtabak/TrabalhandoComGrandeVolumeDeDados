from   sklearn.ensemble import RandomForestClassifier
from   sklearn.ensemble import GradientBoostingClassifier
from   sklearn.svm      import SVC
from   sklearn.tree     import DecisionTreeClassifier
from   sklearn.model_selection    import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy            as np
import csv

#3961346

def readDataset (datasetName, numberOfColumns, removeHeader = False):
    # Array of arrays. Each array is a column of the dataset
    dataset = []

    # Reading dataset
    csvFile   = open(datasetName, 'rb')
    csvReader = csv.reader(csvFile, delimiter=',')

    if (removeHeader):
        csvReader.next()

    for i in xrange(0,numberOfColumns):
        dataset.append([])

    for row in csvReader:
        col = 0
        for column in row:
            dataset[col].append(float(column))
            col += 1

    return dataset

def classify(alg, performCV = False, cv_folds=5, warm_start = False, iterations = 2):
    if (warm_start == False):
        iterations = 1

    for i in xrange(0, iterations):
        alg.fit(np.transpose(train), target)            

    alg.fit(np.transpose(train), target)
    result = np.transpose(alg.predict_proba(np.transpose(test)))[1]

    if (performCV):
         cv_score = cross_val_score(alg, np.transpose(train), target, cv=cv_folds, scoring='roc_auc')
         print "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score))

    return result

trainingDataset = readDataset('dataset_trabalho3/newTrainingDataset.csv', 240)
testingDataset  = readDataset('dataset_trabalho3/newTestingDataset.csv', 239)

target = trainingDataset[-1:][0]
train  = trainingDataset[0:len(trainingDataset)-1]
test   = testingDataset

# alg    = DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)
# alg    = SVC()
# alg    = RandomForestClassifier(n_estimators=200, n_jobs=8)
# alg    = KNeighborsClassifier(n_neighbors = 2)

alg    = GradientBoostingClassifier(warm_start=True)
result = classify(alg, warm_start = True, iterations = 10)
np.savetxt('resultados/result.csv', result, fmt="%f", delimiter=",")