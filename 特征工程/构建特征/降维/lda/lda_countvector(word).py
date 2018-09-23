"""
将countvector(word)特征降维为lda特征，并将结果保存至本地

"""
from sklearn.decomposition import LatentDirichletAllocation
import pickle
import time

t_start = time.time()

"""=====================================================================================================================
1 countvector(word)特征加载
"""
with open('countvector_word.pkl', 'rb') as f:
	x_train, y_train, x_test = pickle.load(f_tf)

"""=====================================================================================================================
2 特征降维：lda
"""
lda = LatentDirichletAllocation(n_components=200)
x_train = lda.fit_transform(x_train)
x_test = lda.transform(x_test)

"""=====================================================================================================================
3 将lda特征保存至本地
"""
data = (x_train, y_train, x_test)
with open('lda_countvector(word).pkl', 'wb') as f:
	pickle.dump(data, f)

t_end = time.time()
print("共耗时：{}min".format((t_end-t_start)/60))