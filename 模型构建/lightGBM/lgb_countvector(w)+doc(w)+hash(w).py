"""
1.特征：countvector(w)+doc(w)+hash(w)
2.模型：lgb
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import lightgbm as lgb


"""=====================================================================================================================
1 读取数据,并转换到lgb的标准数据格式
"""
with open('countvector(w)+doc(w)+hash(w).pkl', 'rb') as f:
    x_train, y_train, x_test = pickle.load(f)

"""划分训练集和验证集，验证集比例为test_size"""
x_train, x_vali, y_train, y_vali = train_test_split(x_train, y_train, test_size=0.1, random_state=0)
d_train = lgb.Dataset(data=x_train, label=y_train)
d_vali = lgb.Dataset(data=x_vali, label=y_vali)

"""=====================================================================================================================
2 训练lgb分类器
"""
params = {
        'boosting': 'gbdt',
        'application': 'multiclassova',
        'num_class': 20,
        'learning_rate': 0.1,
        'num_leaves':31,
        'max_depth':-1,
        'lambda_l1': 0,
        'lambda_l2': 0.5,
        'bagging_fraction' :1.0,
        'feature_fraction': 1.0
        }

bst = lgb.train(params, d_train, num_boost_round=800, valid_sets=d_vali,feval=f1_score_vali, early_stopping_rounds=None,
                verbose_eval=True)
 
"""=====================================================================================================================
3 对测试集进行预测;将预测结果转换为官方标准格式；并将结果保存至本地
"""
y_proba = bst.predict(x_test)
y_test = np.argmax(y_proba, axis=1) + 1

df_result = pd.DataFrame(data={'id':range(102277), 'class': y_test.tolist()})
df_proba = pd.DataFrame(data={'id':range(102277), 'proba': y_proba.tolist()})

df_result.to_csv('lgb_countvector(w)+doc(w)+hash(w).csv',index=False)
df_proba.to_csv('lgb_countvector(w)+doc(w)+hash(w)_proba.csv',index=False)




