from   sklearn.ensemble        import RandomForestClassifier
from   sklearn.ensemble        import GradientBoostingClassifier
from   sklearn.svm             import SVC
from   sklearn.tree            import DecisionTreeClassifier
from   sklearn.model_selection import cross_val_score
from   sklearn.neighbors       import KNeighborsClassifier
from   scipy.optimize          import basinhopping
import numpy                   as np
import csv
import random
import math

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

ranges = np.array([[100,500], [0.01, 0.1], [2,5]])

#Funcao de perturbacao
def perturb(args):
    print("estou na funcao perturb")
    rand = random.randint(0,2)
    print rand
    if (rand == 0):
        args[0] = random.randint(ranges[0][0], ranges[0][1])
    elif (rand == 1):
        args[1] = random.uniform(ranges[1][0], ranges[1][1])
    elif (rand == 2):
        args[2] = random.randint(ranges[2][0], ranges[2][1])
    print args
    return args

#Funcao de Aceitacao
def accept(f_new, x_new, f_old, x_old):
    print("estou na funcao accept")
    return f_new < f_old

#Funcao Objetivo
def classify(args, performCV = False):
    print("estou na funcao classify")
    print args
    _n_estimators  = args[0]
    _learning_rate = args[1]
    _max_depth     = args[2]
    alg = GradientBoostingClassifier(n_estimators=int(_n_estimators), learning_rate=float(_learning_rate),max_depth=int(_max_depth))
    alg.fit(np.transpose(train), target)

    # result = np.transpose(alg.predict_proba(np.transpose(test)))[1]
    cv_score = cross_val_score(alg, np.transpose(train), target, cv=5, scoring='roc_auc')
    print "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score))
    return 1 - cv_score.mean()

def simulatedAnnealing(args, t0, niter, nper, n_sucess_iter, alpha) :
    print "incio da funcao"
    arq = open('resultados/simulatedAnnealing.txt', 'w')
    s0 = np.array([200, 0.1, 3])
    s = s0
    f = classify(s)
    t = t0
    for j in xrange(niter):
        print "incio da j"
        nSucesso = 0
        for i in xrange(nper):
            print "incio da i"
            if(nSucesso >= n_sucess_iter):
                break
            si = perturb(s)
            fnew = classify(si)
            deltaFi = fnew - f
            print deltaFi
            if(deltaFi <= 0 or math.exp(- deltaFi / t ) > random.uniform(0,1)):
                s = si
                nSucesso = nSucesso + 1
                f = fnew
                arq.write(str(s))
                arq.write(",")
                arq.write(str(f))
                arq.write('\n')
                arq.flush()
        t = alpha * t
        if(nSucesso == 0):
            break
    print s


trainingDataset = readDataset('dataset_trabalho3/newTrainingDataset.csv', 240)
testingDataset  = readDataset('dataset_trabalho3/newTestingDataset.csv', 239)

target = trainingDataset[-1:][0]
train  = trainingDataset[0:len(trainingDataset)-1]
test   = testingDataset

# alg    = DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)
# alg    = SVC()
# alg    = RandomForestClassifier(n_estimators=200, n_jobs=8)
# alg    = KNeighborsClassifier(n_neighbors = 2)

#n_estimators = [100,1000]

#result =classify([200,0.1,3])
x0 = np.array([200, 0.1, 3])
# ret = basinhopping(classify, x0, minimizer_kwargs={"method":"BFGS"}, take_step=perturb, niter=20)
simulatedAnnealing(x0, 1000, 20, 100, 20, 0.5)

# print perturb(x0)

# print("global minimum: x = [%.4f, %.4f], f(x0) = %.4f" % (ret.x[0],ret.x[1],ret.fun))
#np.savetxt('resultados/result.csv', result, fmt="%f", delimiter=",")
