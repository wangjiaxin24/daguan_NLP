'''
用lr从tfidf(word)中挑选特征，并将结果保存到本地

tfidf(article)可做类似处理

'''

import time
import pickle
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

t_start = time.time()

"""读取tfidf(word)特征"""
with open('tfidf_word.pkl', 'rb') as fp:
    x_train, y_train, x_test = pickle.load(fp)

"""进行特征选择"""

LR = LogisticRegression(C=120, dual=False).fit(x_train, y_train)
slt = SelectFromModel(LR, prefit=True)
x_train_s = slt.transform(x_train)
x_test_s = slt.transform(x_test)

"""保存选择后的特征至本地"""
num_features = x_train_s.shape[1]

with open('lr-tfidf(word).pkl', 'wb') as data_f:
    pickle.dump((x_train_s, y_train, x_test_s), data_f)

t_end = time.time()
print("特征选择完成，选择{}个特征，共耗时{}min".format(num_features, (t_end-t_start)/60))