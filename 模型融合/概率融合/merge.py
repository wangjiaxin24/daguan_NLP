#import pickle
import pandas as pd
import numpy as np
import time


path='D:/daguanbei/data/proba'

print('1 读取概率+投票')
svm_1 = pd.read_csv(path+'/result_proba_svm_0.779.csv')
svm_2 = pd.read_csv(path+'/sfm_lr_lsvm(C5)_a_proba_0.779.csv')
svm_3 = pd.read_csv(path+'/sfm_lr_lsvm(C5)_w_proba_0.779.csv')

lr_1 = pd.read_csv(path+'/result_proba_lr_0.777.csv')
lr_2 = pd.read_csv(path+'/LR_LR_selectfeature_article_prob_0.777.csv')


lgb_1 = pd.read_csv(path+'/lgb_article_merge(tf_doc_hash)_0.772.csv')
lgb_2 = pd.read_csv(path+'/lgb_word_merge(tf_doc_hash)_0.773.csv')

bys_1 = pd.read_csv(path+'/bayes_tfidf_w_prob_0.72.csv')

# 辅助函数
def series2arr(series):
    res = []
    for row in series:
        res.append(np.array(eval(row)))
    return np.array(res)


svm_1 = series2arr(svm_1['proba'])
svm_2 = series2arr(svm_2['proba'])
svm_3 = series2arr(svm_3['proba'])


lr_1 = series2arr(lr_1['proba'])
lr_2 = series2arr(lr_2['proba'])


lgb_1 = series2arr(lgb_1['proba'])
lgb_2 = series2arr(lgb_2['proba'])

bys_1 = series2arr(bys_1['proba'])

# final_prob = 2*svm_prob_arr+lg_prob_arr
model_list = [svm_1,svm_2,svm_3,svm_4,svm_5,svm_6,lr_1,lr_2,lr_3,lr_4,lr_5,lgb_1,lgb_2,bys_1]
final_prob = 0.0
for i in model_list:
	final_prob += i


y_class=[np.argmax(row)+1 for row in final_prob]
df_result = pd.DataFrame({'id':range(102277),'class':y_class})

df_result.to_csv('merge.csv',index=False)