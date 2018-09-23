"""
将tfidf(word)和tfidf(article)拼接成新的特征

"""

import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

"""=====================================================================================================================
1 数据预处理
"""
read_start_time = time.time()
df_train=pd.read_csv('train_set.csv')
df_test=pd.read_csv('test_set.csv')


#df_train.drop(df_train.columns[0],axis=1,inplace=True)

df_train["word_article"] = df_train["article"].map(str) +' '+ df_train["word_seg"].map(str)
df_test["word_article"] = df_test["article"].map(str) +' ' + df_test["word_seg"].map(str)
y_train = (df_train['class'] - 1).values

"""=====================================================================================================================
2 特征工程
"""
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True)
vectorizer.fit(df_train['word_article'])
x_train = vectorizer.transform(df_train['word_article'])
x_test = vectorizer.transform(df_test['word_article'])

"""=====================================================================================================================
3 保存至本地
"""
data = (x_train, y_train, x_test)
with open('./tfidf(word+article).pkl', 'wb') as f：
	pickle.dump(data, f)
