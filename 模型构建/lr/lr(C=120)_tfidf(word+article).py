"""
1.特征：tfidf(word+article)
2.模型：lr
3.参数：C=120
"""

import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

with open('tfidf(word+article).pkl', 'rb') as f:
    x_train, y_train, x_test = pickle.load(f)

clf = LogisticRegression(C=120, dual=False)
clf.fit(x_train, y_train)
 
#返回预测标签
y_test = clf.predict(x_test)
y_test_prob = clf.predict_proba(x_test)


#标签预测结果存储
y_test = [i+1 for i in y_test_list.tolist()]
y_test_prob = y_test_prob_LR.tolist()

df_result = pd.DataFrame({'id':range(102277), 'class': y_test})
df_proba = pd.DataFrame({'id':range(102277), 'prob': y_test_prob})

df_result.to_csv('lr(C=120)_tfidf(word+article).csv',index=False)
df_proba.to_csv('lr(C=120)_tfidf(word+article)_proba.csv',index=False)
