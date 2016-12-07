import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np

print("1")
# Load the data
train_df = pd.read_csv('dataset_trabalho3/train_file.csv', header=0)
test_df  = pd.read_csv('dataset_trabalho3/test_file.csv',  header=0)

header = list(train_df.columns.values)

complete_dataframe = train_df[header[:-1]].append(test_df[header[:-1]])

print("2")
train_X = complete_dataframe[0:train_df.shape[0]].as_matrix()
test_X  = complete_dataframe[train_df.shape[0]::].as_matrix()
train_y = train_df['TARGET']

print("3")
#127,0.0522649127,5
gbm = xgb.XGBClassifier(max_depth=5, n_estimators=127, learning_rate=0.0522649127).fit(train_X, train_y)
predictions = gbm.predict_proba(test_X)
print("4")

result = []
for i in xrange(0, len(predictions)):
	print(predictions[i][1])
	result.append(predictions[i][1])

# submission = pd.DataFrame({ 'TARGET': predictions })
# submission.to_csv("submission.csv", index=False)
np.savetxt('resultados/xgb_result.csv', np.array(result), fmt="%f", delimiter=",")
print("5")