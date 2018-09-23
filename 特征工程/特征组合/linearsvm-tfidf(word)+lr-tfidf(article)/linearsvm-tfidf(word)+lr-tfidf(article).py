"""
将linearsvm挑选的tfidf(word)特征和lr挑选的tfidf(article)

"""

import pickle
from scipy import sparse
from scipy.sparse import hstack

with open('linearsvm-tfidf(word).pkl', 'rb')as f_1:
    x_train_1, y_train, x_test_1 = pickle.load(f_1)


with open('lr-tfidf(article).pkl', 'rb')as f_2:
    x_train_2, y_train, x_test_2 = pickle.load(f_2)
    

x_train = hstack([x_train_1, x_train_2])
x_test = hstack([x_test_1, x_test_2])

data = (x_train, y_train, x_test)
with open('linearsvm-tfidf(word)+lr-tfidf(article).pkl', 'wb')as f:
    pickle.dump(data, f)
