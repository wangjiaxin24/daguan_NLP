"""
将countvector(word)、hash(word)和doc2vec(word)拼接成新特征

"""
import pickle
from scipy import sparse
from scipy.sparse import hstack

"""读取hash(word)和doc2vec(word)特征"""
with open('./doc2vec_word.pkl', 'rb') as f_1:
	x_train_1, y_train, x_test_1 = pickle.load(f_1)

with open('./hash_word.pkl', 'rb') as f_2:
	x_train_2, _, x_test_2 = pickle.load(f_2)

"""将numpy 数组 转换为 csr稀疏矩阵"""
x_train_1 = sparse.csr_matrix(x_train_1)
x_test_1 = sparse.csc_matrix(x_test_1)

x_train_2 = sparse.csr_matrix(x_train_2)
x_test_2 = sparse.csc_matrix(x_test_2)

"""读取tfidf(word)特征"""
with open('./tfidf_word.pkl', 'rb') as f_3:
	x_train_3, _, x_test_3= pickle.load(f_3)

"""对两个稀疏矩阵进行合并"""
x_train_4 = hstack([x_train_1, x_train_2])
x_test_4 = hstack([x_test_1, x_test_2])

x_train_5 = hstack([x_train_4, x_train_3])
x_test_5 = hstack([x_test_4, x_test_3])

"""将合并后的稀疏特征保存至本地"""
data = (x_train_5, y_train, x_test_5)
with open('./countvector(w)+doc(w)+hash(w).pkl', 'wb') as f:
	pickle.dump(data, f)





