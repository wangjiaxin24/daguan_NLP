'''
将原始数据的word特征数字化为doc2vec特征，并将结果保存到本地

article特征可做类似处理

'''
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import time
import pickle

t_start = time.time()

"""=====================================================================================================================
0 辅助函数 
"""

def sentence2list(sentence):
    s_list = sentence.strip().split() #strip()去掉首尾空格，split()将字符串以空格切分成列表
    return s_list

"""=====================================================================================================================
1 加载原始数据
"""
df_train=pd.read_csv('train_set.csv')
df_test=pd.read_csv('test_set.csv')

df_train.drop(columns='article', inplace=True)
df_test.drop(columns='article', inplace=True)

# 按行拼接df_train和df_test
df_all = pd.concat(objs=[df_train, df_test], axis=0, sort=True)

# 获取train文件中的特征class
y_train = (df_train['class'] - 1).values

df_all['word_list'] = df_all['word_seg'].apply(sentence2list)
texts = df_all['word_list'].tolist()

"""=====================================================================================================================
2 特征工程
"""
print('2 特征工程')
# 将原始数据数字化为doc2vec

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]
model = Doc2Vec(documents, vector_size=200, window=5, min_count=3, workers=4, epochs=25)
docvecs = model.docvecs

x_train = []
for i in range(0, 102277):
    x_train.append(docvecs[i])
x_train = np.array(x_train)

x_test = []
for j in range(102277, 204554):
    x_test.append(docvecs[j])
x_test = np.array(x_test)

"""=====================================================================================================================
3 保存至本地
"""
print('3 保存特征')
data = (x_train, y_train, x_test)

with open('doc2vec_word.pkl', 'wb') as f:
	pickle.dump(data,f) 

t_end = time.time()
print("共耗时：{}min".format((t_end-t_start)/60))
# 共耗时：54min