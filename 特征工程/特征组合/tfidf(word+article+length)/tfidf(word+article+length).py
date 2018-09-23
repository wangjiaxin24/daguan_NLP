'''
特征拼接，拼接文章长度
#1.载入特征
#2 读文章长度
#3 特征缩放，拼接特征
'''

import pickle

# 载入特征
with open('tfidf(word+article).pkl','rb') as f:
	x_train,y_train,y_test = pickle.load(f)


# 读取文章长度信息
import pandas as pd 
import numpy as np 
from sklearn import preprocessing

# 获取x_train文件的article和word的长度
train_article = pd.read_csv('train_article_len.csv')
train_word = pd.read_csv('train_word_len.csv')
train_article_len = train_article['article_len']
train_word_len = train_word['word_len']

# 获取x_test文件的article和word的长度
test_article= pd.read_csv('test_article_len.csv')
test_word = pd.read_csv('test_word_len.csv')
test_article_len = test_article['article_len']
test_word_len = test_word['word_len']


# 特征缩放
# 将x_train article和word长度缩放到0-1区间
# 将x_test article和word长度缩放到0-1区间
# np.c_按行连接两个矩阵，就是把两个矩阵左右相加
train_len = np.c_[train_article_len.values, train_word_len.values]  
test_len = np.c_[test_article_len.values, test_word_len.values]
min_max_scaler = preprocessing.MinMaxScaler()            
train_len= min_max_scaler.fit_transform(train_len)
test_len= min_max_scaler.fit_transform(test_len)


# 获取article,word和len拼接后的特征feature_c_train和feature_c_test
from scipy.sparse import coo_matrix, hstack,vstack 

def concat(a,b):
	row = np.array(range(a.shape[0]))
	col = np.array([0]*a.shape[0])
	data = b['word_len'].values
	b = csr_matrix((data, (row, col)), shape=(a.shape[0], 1))

	res = hstack((a,b))
	return res.tocsr()

feature_c_train = concat([x_train,train_len])
feature_c_test= concat([x_test,test_len])

# 保存特征
with open('保存地址/tfidf(word+article+length).pkl', 'wb') as f:
    pickle.dump((feature_c_train, y_train, feature_c_test),  f)