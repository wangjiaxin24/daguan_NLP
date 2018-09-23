"""
1.特征：linearsvm-tfidf(word)+lr-tfidf(article) / doc2vec_word
2.模型：bayes

"""
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import pickle
import time
import random
from scipy.stats import randint as sp_randint
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,BaseDiscreteNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import scipy


def get_time_diff(start_time,str):
    print("%s 用时:%f min" % (str,(time.time() - start_time)/60))
    return time.time() - start_time

DATA_PATH = '..\达观杯_特征\\feature_word\Doc2vec_word\Doc2vec_word\data_doc2vec.pkl'
SAVE_PATH = './models/bayes/HashingVectorizer_word'


def read_data(path):
    with open(path, 'rb') as f:
        x_train, y_train, x_test = pickle.load(f)
    X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.25)
    return X_train, X_val, Y_train, Y_val, x_test


def model_search_random(X_train, Y_train, param_dist, n_iter_search = 20,clf = MultinomialNB()):
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search)
    random_search.fit(X_train, Y_train)
    return random_search


def model_search_grid(X_train, Y_train, param_dist,clf = MultinomialNB()):
    random_search = GridSearchCV(estimator=clf,param_grid=param_dist)
    random_search.fit(X_train, Y_train)
    return random_search


def model_best(model_search):
    print('随机搜索-度量记录：', model_search.cv_results_)  # 包含每次训练的相关信息
    print('随机搜索-最佳度量值:', model_search.best_score_)  # 获取最佳度量值
    print('随机搜索-最佳参数：', model_search.best_params_)  # 获取最佳度量值时的代定参数的值。是一个字典
    print('随机搜索-最佳模型：', model_search.best_estimator_)  # 获取最佳度量时的分类器模型
    return model_search.cv_results_, model_search.best_score_,model_search.best_params_,model_search.best_estimator_


def predict(X_val,Y_val,x_test,model):
    Y_val_test = model.predict(X_val)
    score = accuracy_score(Y_val, Y_val_test)
    y_test = model.predict(x_test)
    y_test = [i + 1 for i in y_test.tolist()]
    df_result = pd.DataFrame({'id': range(102277), 'class': y_test})
    y_test_proba = model.predict_proba(x_test)
    y_test_proba = y_test_proba.tolist()
    df_proba = pd.DataFrame({'id': range(102277), 'proba': y_test_proba})
    return df_result,df_proba,score


def save(score,path_save,df_result,df_proba,model,alpha = None):
    if alpha is not None:
        class_name = '/bayes_%0.2f_%s_class_%0.4f.csv' % (alpha,"data_tf_word",score)
        proba_name = '/bayes_%0.2f_%s_proba_%0.4f.csv' % (alpha,"data_tf_word",score)
        model_name = '/bayes_%0.2f_%s_model_%0.4f.csv' % (alpha,"data_tf_word",score)
    else:
        class_name = '/bayes_%s_class_%0.4f.csv' % ("data_tf_word",score)
        proba_name = '/bayes_%s_proba_%0.4f.csv' % ("data_tf_word",score)
        model_name = '/bayes_%s_model_%0.4f.csv' % ("data_tf_word",score)
    with open(path_save + model_name, 'wb')as f:
        pickle.dump(model, f)
    df_result.to_csv(path_save + class_name, index=False)
    df_proba.to_csv(path_save + proba_name, index=False)


def test():
    _start = time.time()
    __start = _start # 记录总的时间
    X_train, X_val, Y_train, Y_val, x_test = read_data(DATA_PATH)
    get_time_diff(_start,"读取数据")

    _start = time.time()
    params = {"alpha": np.arange(0.75,0.86,0.01)}
    # search_model = model_search_random(X_train,Y_train,param_dist=params,n_iter_search=15,clf=MultinomialNB())
    search_model = model_search_grid(X_train, Y_train, param_dist=params,clf = MultinomialNB())
    get_time_diff(_start,"模型搜索")

    _start = time.time()
    cv_results_, best_score_, best_params_, model_best_estimator_ = model_best(search_model)
    alpha = best_params_['alpha']
    # fit_prior = best_params_['fit_prior']
    model = CalibratedClassifierCV(base_estimator=MultinomialNB(alpha=alpha))
    model.fit(X_train,Y_train)
    get_time_diff(_start,"最佳模型")

    _start = time.time()
    df_res, df_prob, score = predict(X_val,Y_val,x_test,model)
    get_time_diff(_start, "预测结果")
    print("验证集分数：%0.4f"%score)
    _start = time.time()
    save(score,SAVE_PATH,df_res,df_prob,model,alpha=alpha)
    get_time_diff(_start,"保存模型")
    get_time_diff(__start,"共用")
    

if __name__ == '__main__':
       test()
