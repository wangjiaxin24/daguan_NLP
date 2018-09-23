# -*- coding: utf-8 -*-
"""
用tfidf(word+article)特征训练linear模型

"""
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression






with open('tfidf(word+article).pkl' , 'rb') as f:
    x_train, y_train, x_test = pickle.load(f)

#模型构建和训练
clf = LinearRegression(n_jobs = -1)

clf.fit(x_train, y_train)
 
#返回预测标签
y_test = clf.predict(x_test)


#标签预测结果存储
y_test_list = y_test.tolist()
df_test = [i+1 for i in y_test_list]
df_result = pd.DataFrame({'id':range(102277), 'class': df_test})
df_result.to_csv('linear_tfidf(word+article).csv',index=False)
 
#模型存储
with open('linear_tfidf(word+article).pkl', 'wb') as f:
      pickle.dump(clf, f)


#返回预测属于某标签的概率
y_test_proba_linear = clf.predict_proba(x_test)

#概率值存储
y_test_proba_linear = y_test_proba_linear.tolist()
y_test_proba_linear = pd.DataFrame({'id':range(102277), 'proba': y_test_prob_linear})
y_test_proba_linear.to_csv('linear_tfidf(word+article)_proba.csv',index=False)

