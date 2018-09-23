"""
模型：xgboost
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import time
import joblib

start_time = time.time()

# df_train = pd.read_csv('H:/Datas/new_data/train_set.csv')
df_test = pd.read_csv('H:/Datas/new_data/test_set.csv')

xgb = XGBClassifier(
    learning_rate=0.2,
    n_estimators=60,
    max_depth=4,
    min_child_weight=6,
    gamma=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softmax',
    nthread=4,
    scale_pos_weight=1,
    n_jobs=-1,
    seed=27)

x_train, y_train, x_test = joblib.load('merge_wore_tf_doc_hash.pkl')
# x_train_t, x_test_t, y_train_t, y_test_t = train_test_split(x_train, y_train, test_size=0.3, random_state=66)

xgb.fit(x_train, y_train)


y_test = xgb.predict(x_test)
y_test_proba = xgb.predict_proba(x_test)[:, 1]
joblib.dump(y_test_proba, 'y_test_proba.pkl')
# print("Accuracy : %.4g" % metrics.accuracy_score(y_test_t, y_test))
# print("AUC Score (Train): ", metrics.roc_auc_score(y_test_t, y_test_proba))



df_test['class'] = y_test.tolist()
df_test['class'] = df_test['class'] + 1

df_result = df_test.loc[:, ['id', 'class']]
df_result.to_csv('H:/Datas/new_data/test_set_xgb.csv', index=False)

print("总运行时长：", time.time() - start_time)
