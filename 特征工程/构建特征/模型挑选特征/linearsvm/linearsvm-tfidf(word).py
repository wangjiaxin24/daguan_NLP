"""
用linearsvm从tfidf(word)中挑选特征，并将结果保存到本地

tfidf(article)可做类似处理

"""

import time
import pickle
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC

t_start = time.time()

"""读取特征"""

with open('tfidf_word.pkl', 'rb') as f:
	x_train, y_train, x_test = pickle.load(f)

"""进行特征选择"""
lsvc = LinearSVC(C=0.5, dual=False).fit(x_train, y_train)
slt = SelectFromModel(lsvc, prefit=True)
x_train_s = slt.transform(x_train)
x_test_s = slt.transform(x_test)

"""保存选择后的特征至本地"""
num_features = x_train_s.shape[1]

with open('linearsvm-tfidf(word).pkl', 'wb') as f:
	pickle.dump((x_train_s, y_train, x_test_s), data_f)

t_end = time.time()
print("特征选择完成，选择{}个特征，共耗时{}min".format(num_features, (t_end-t_start)/60))

# 特征选择完成，选择888357个特征，共耗时11.78min