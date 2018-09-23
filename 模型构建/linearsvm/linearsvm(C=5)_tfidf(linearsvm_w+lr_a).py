"""
1.特征：linearsvm-tfidf(word)+lr-tfidf(article)
2.模型：linearsvm
3.参数：C=5
"""

from sklearn.svm import LinearSVC # 支持向量机
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
import time
import pickle


time_start = time.time()
# 读取特征
with open('linearsvm-tfidf(word)+lr-tfidf(article).pkl','rb') as f:
    x_train, y_train, x_test = pickle.load(f)
# 构建模型
clf = CalibratedClassifierCV(base_estimator=LinearSVC(C=5))
clf.fit(x_train, y_train)

# 保存模型
with open('linearsvm(C=5)_tfidf(linearsvm_w+lr_a).pkl','wb') as f:
    pickle.dump(clf,f)
#或者用 joblib.dump(clf, "文件名.pkl") 保存模型

# 预测结果：分类结果和概率结果
y_test = clf.predict(x_test)
y_test_proba = clf.predict_proba(x_test)


# 保存模型输出的分类文件和概率文件
y_test = [i+1 for i in y_test.tolist()]
y_test_proba = y_test_proba.tolist()

df_result = pd.DataFrame({'id':range(102277),'class':y_test})
df_proba = pd.DataFrame({'id':range(102277),'proba':y_test_proba})


df_result.to_csv('./ls(C=5)_tfidf(ls_w+lr_a).csv',index=False)
df_proba.to_csv('./ls(C=5)_tfidf(ls_w+lr_a)_proba.csv',index=False)

time_end = time.time()
print('共耗时：{:.2f}min'.format((time_end-time_start)/60))