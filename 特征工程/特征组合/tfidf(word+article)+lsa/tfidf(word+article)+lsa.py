"""
将tfidf(word+article)特征降维为lsa特征，并将结果保存至本地，并将结果保存到本地

"""
from sklearn.decomposition import TruncatedSVD
import pickle
import time

t_start = time.time()


"""=====================================================================================================================
0 读取tfidf(word+article)特征
"""
with open('tfidf(word+article).pkl.pkl', 'rb') as f:
	x_train, y_train, x_test = pickle.load(f)

"""=====================================================================================================================
1 特征降维：lsa
"""
lsa = TruncatedSVD(n_components=200)
x_train = lsa.fit_transform(x_train)
x_test = lsa.transform(x_test)

"""=====================================================================================================================
2 将lsa特征保存至本地
"""
data = (x_train, y_train, x_test)
with open('tfidf(word+article)+lsa.pkl', 'wb') as f:
	pickle.dump(data, f_data)

t_end = time.time()
print("共耗时：{}min".format((t_end-t_start)/60))