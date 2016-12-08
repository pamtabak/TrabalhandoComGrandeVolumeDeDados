import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np

print("loading the data")
# Load the data
train_df = pd.read_csv('dataset_trabalho3/train_without_correlation095.csv', header=0)
test_df  = pd.read_csv('dataset_trabalho3/test_without_correlation095.csv',  header=0)

y = np.array(train_df["TARGET"])
train_df.drop(['TARGET'], axis=1, inplace=True)
X = np.array(train_df)

test_np  = np.array(test_df)

dtrain_matrix = xgb.DMatrix(X, y)
dtest_matrix  = xgb.DMatrix(test_np)

param = {'max_depth':5, 'eta':0.0522649127, 'objective':'binary:logistic', 'eval_metric':'auc', 'silent':1, 'min_child_weight':1}
num_round = 200
print("training")
bst = xgb.train(param, dtrain_matrix, num_round)

print("predicting")
result = bst.predict(dtest_matrix)
np.savetxt('resultados/xgb_result.csv', np.array(result), fmt="%f", delimiter=",")

# score = xgb.cv(param, dtrain_matrix, num_round, nfold=5, metrics={'auc'})
# print(score)