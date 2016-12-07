import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np

print("loading the data")
# Load the data
train_df = pd.read_csv('dataset_trabalho3/train_file.csv', header=0)
test_df  = pd.read_csv('dataset_trabalho3/test_file.csv',  header=0)

header = list(train_df.columns.values)

complete_dataframe = train_df[header[:-1]].append(test_df[header[:-1]])

train_X = complete_dataframe[0:train_df.shape[0]].as_matrix()
test_X  = complete_dataframe[train_df.shape[0]::].as_matrix()
train_y = train_df['TARGET']

print("classifing")
gbm = xgb.XGBClassifier(max_depth=5, n_estimators=127, learning_rate=0.0522649127).fit(train_X, train_y)
print("predicting")
predictions = gbm.predict_proba(test_X)

print("cleaning results")
#We only want the probability of being from class 1
result = []
for i in xrange(0, len(predictions)):
	print(predictions[i][1])
	result.append(predictions[i][1])

print("saving results")
np.savetxt('resultados/xgb_result.csv', np.array(result), fmt="%f", delimiter=",")
print("finished")