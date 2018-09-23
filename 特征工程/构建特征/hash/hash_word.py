'''
将原始数据的word特征数字化为hash特征，并将结果保存到本地

article特征可做类似处理

'''
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
import pickle
import time

t_start = time.time()

"""=====================================================================================================================
1 加载原始数据
"""
# 读取原始数据train和test文件
df_train=pd.read_csv('train_set.csv')
df_test=pd.read_csv('test_set.csv')

# 删除特征article，只保留特征word
df_train.drop(columns='article', inplace=True)
df_test.drop(columns='article', inplace=True)

# 按行拼接df_train和df_test
df_all = pd.concat(objs=[df_train, df_test], axis=0, sort=True)

# 获取train文件中的特征class
y_train = (df_train['class'] - 1).values

"""=====================================================================================================================
2 特征工程
"""
print('2 特征工程')
# 将原始数据数字化为hash特征

vectorizer = HashingVectorizer(ngram_range=(1, 2), n_features=200)
d_all = vectorizer.fit_transform(df_all['word_seg'])
x_train = d_all[:len(y_train)]
x_test = d_all[len(y_train):]

"""=====================================================================================================================
3 保存至本地
"""
print('3 保存特征')
data = (x_train.toarray(), y_train, x_test.toarray())
with open('hash_word.pkl', 'wb') as f:
	pickle.dump(data,f)
	
t_end = time.time()
print("共耗时：{}min".format((t_end-t_start)/60))
# 共耗时：4.8min