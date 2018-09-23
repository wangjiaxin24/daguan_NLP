# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 12:17:36 2018

@author: 52
"""

#导模型包
#导入其他包
#import numpy as np
import pandas as pd
import random
import math
#from sklearn import metrics
#from random import randint
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


# 染色体长度 ，需要进行二进制转换
#learning_rate: [0.01, 0.02, 0.03, 0.04, 0.05, ...... 0.19, 0.20]   0.2/0.01=20份，用5位二进制表示足够（2的4次方<20<2的5次方）
#     00000 -----> 0.01
#    11111 -----> 0.20
#   0.01 + 对应十进制*（0.20-0.01）/ (2的5次方-1)
#max_depth:[3、4、5、6、7、8、9、10]   用3位二进制
#num_leaves: [10、 20、 30、......150、160] 用4位二进制, 0000代表10
#5+3+4=12

chrom_length = 12    
cons_value = 0.000019 / 31 # (0.20-0.01）/ (32 - 1) #learning_rate的转换系数

#GA模型初始参数
max_iter=10

tol=4.064516129032258e-06

C=6



generations = 20   # 繁殖代数
pop_size = 10      # 种群数量  
# max_value = 10      # 基因中允许出现的最大值 （可防止离散变量数目达不到2的幂的情况出现，限制最大值，此处不用） 
pc = 0.6            # 交配概率  
pm = 0.01           # 变异概率  
results = []      # 存储每一代的最优解，N个三元组（auc最高值, n_estimators, max_depth）  
fit_value = []      # 个体适应度  
fit_mean = []       # 平均适应度 
random_seed = 20     #随机种子


#数据读取
df_train1 = pd.read_csv('train_set0.csv')

df_train1.drop(columns=['article','id'],inplace=True)


#特征提取模型的训练
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9, use_idf=1,smooth_idf=1, sublinear_tf=1)
vectorizer.fit(df_train1['word_seg'])

print(0)
#特征提取模型数据处理
df_train = pd.read_csv('train_set.csv')
df_train.drop(columns=['article','id'],inplace=True)

x_train = vectorizer.transform(df_train['word_seg'])

y_train = df_train['class']-1




#读取处理切分数据集，训练模型并输出auc分数
def lgbModel(max_iter,tol, C, random_seed, x_train, y_train):

    x_train1,x_test1,y_train1,y_test1 = train_test_split(x_train,y_train,test_size=0.4)
    
    model = LogisticRegression(penalty='l1',dual=False, solver='liblinear', tol=tol,  C=C, max_iter=max_iter, multi_class='ovr', verbose=1)
    model.fit(x_train1,y_train1)
    
    predict_y = model.predict(x_test1)
    roc_auc =  f1_score(y_test1, predict_y, average='micro')
    
    return roc_auc 
 #scoring='roc_auc'是用来检测定性数据结果的，比如好人坏人，是和否等，即结果为0或1的数据预测。
 
 
# Step 1 : 对参数进行编码（用于初始化基因序列，可以选择初始化基因序列，本函数省略）
def geneEncoding(pop_size, chrom_length):  
    pop = [[]]
    for i in range(pop_size):
        temp = []
        for j in range(chrom_length):
            temp.append(random.randint(0, 1))
        pop.append(temp)
    return pop[1:]

# 对每个个体进行解码，并拆分成单个变量，返回 
def decodechrom(pop):
    variable = []
    for i in range(len(pop)):
        res = []
        
        # 计算第一个变量值，即 0101->10(逆转)
        temp1 = pop[i][0:4]
        v1 = 0
        for i1 in range(4):
            v1 += temp1[i1] * (math.pow(2, i1))
        res.append(int(v1))
        
        # 计算第二个变量值
        temp2 = pop[i][4:9]
        v2 = 0
        for i2 in range(5):
            v2 += temp2[i2] * (math.pow(2, i2))
        res.append(int(v2))
 
        # 计算第三个变量值
        temp3 = pop[i][9:12]
        v3 = 0
        for i3 in range(3):
            v3 += temp3[i3] * (math.pow(2, i3))
        res.append(int(v3))
 
        variable.append(res)
    return variable



# Step 2 : 计算个体的目标函数值
def cal_obj_value(pop):
    objvalue = []
    variable = decodechrom(pop)
    for i in range(len(variable)):
        tempVar = variable[i]
 
        max_iter = (tempVar[0] + 1)* 10
        print(max_iter)
        tol = 0.000001 + tempVar[1] * cons_value
        print(tol)
        C = 3 + tempVar[2]
        print(C)
        
        aucValue = lgbModel(max_iter,tol, C, random_seed, x_train, y_train)
        print(aucValue)
        objvalue.append(aucValue)
    return objvalue #目标函数值objvalue[m] 与个体基因 pop[m] 对应 
 
 
 
# Step 3: 计算个体目标函数值，就是适应度值（淘汰负值）
def calfitvalue(obj_value):
    fit_value = []
    temp = 0.0
    Cmin = 0
    for i in range(len(obj_value)):
        if(obj_value[i] + Cmin > 0):
            temp = Cmin + obj_value[i]
        else:
            temp = 0.0
        fit_value.append(temp)
    return fit_value

 
# Step 4: 找出适应函数值中最大值，和对应的个体
def best(pop, fit_value):
    best_individual = pop[0]
    best_fit = fit_value[0]
    for i in range(1, len(pop)):
        if(fit_value[i] > best_fit):
            best_fit = fit_value[i]
            best_individual = pop[i]
    return [best_individual, best_fit]
 
 
# Step 5: 每次繁殖，将最好的结果记录下来(将二进制转化为十进制)
def b2d(best_individual):
    # 计算第一个变量值
    temp1 = best_individual[0:4]
    v1 = 0
    for i1 in range(4):
        v1 += temp1[i1] * (math.pow(2, i1))
    v1 = (v1 + 1) * 10
    
    # 计算第二个变量值
    temp2 = best_individual[4:9]
    v2 = 0
    for i2 in range(5):
        v2 += temp2[i2] * (math.pow(2, i2))
    v2 = 0.000001 + v2 * cons_value
 
    # 计算第三个变量值
    temp3 = best_individual[9:12]
    v3 = 0
    for i3 in range(3):
        v3 += temp3[i3] * (math.pow(2, i3))
    v3 = 3 + v3
 
    return int(v1), float(v2), int(v3)
 
 

# 求适应值的总和
def sum(fit_value):
    total = 0
    for i in range(len(fit_value)):
        total += fit_value[i]
    return total
 
# 计算累积概率
def cumsum(fit_value):
    temp=[]
    for i in range(len(fit_value)):
        t = 0
        j = 0
        while(j <= i):
            t += fit_value[j]
            j = j + 1
        temp.append(t)
    for i in range(len(fit_value)):
        fit_value[i]=temp[i]


# Step 6: 自然选择（轮盘赌算法）
def selection(pop, fit_value):
    # 计算每个适应值的概率
    new_fit_value = []
    total_fit = sum(fit_value)
    for i in range(len(fit_value)):
        new_fit_value.append(fit_value[i] / total_fit)
    # 计算每个适应值的累积概率
    cumsum(new_fit_value)
    # 生成随机浮点数序列
    ms = []
    pop_len = len(pop)
    for i in range(pop_len):
        ms.append(random.random())
    # 对生成的随机浮点数序列进行排序
    ms.sort()
    # 轮盘赌算法（选中的个体成为下一轮，没有被选中的直接淘汰，被选中的个体代替）
    fitin = 0
    newin = 0
    newpop = pop
    while newin < pop_len:
        if(ms[newin] < new_fit_value[fitin]):
            newpop[newin] = pop[fitin]
            newin = newin + 1
        else:
            fitin = fitin + 1
    pop = newpop
 

 
# Step 7: 交叉繁殖
def crossover(pop, pc): #个体间交叉，实现基因交换
    poplen = len(pop)
    for i in range(poplen - 1):
        if(random.random() < pc):
            cpoint = random.randint(0,len(pop[0]))
            temp1 = []
            temp2 = []
            temp1.extend(pop[i][0 : cpoint])
            temp1.extend(pop[i+1][cpoint : len(pop[i])])
            temp2.extend(pop[i+1][0 : cpoint])
            temp2.extend(pop[i][cpoint : len(pop[i])])
            pop[i] = temp1
            pop[i+1] = temp2
 
 
# Step 8: 基因突变
def mutation(pop, pm):
    px = len(pop)
    py = len(pop[0])
    for i in range(px):
        if(random.random() < pm):
            mpoint = random.randint(0, py-1)
            if(pop[i][mpoint] == 1):
                pop[i][mpoint] = 0
            else:
                pop[i][mpoint] = 1
 
 
def writeToFile(var, w_path):
    f=open(w_path,"a+")
    for item in var:
        f.write(str(item) + "\r\n")
    f.close()
 
 
def generAlgo(generations):
    pop = geneEncoding(pop_size, chrom_length)
    print(str(generations)+" start...")
    for i in range(generations):
        # print("第 " + str(i) + " 代开始繁殖......")
        obj_value = cal_obj_value(pop) # 计算目标函数值
        # print(obj_value)
        fit_value = calfitvalue(obj_value) #计算个体的适应值
        # print(fit_value)
        [best_individual, best_fit] = best(pop, fit_value) #选出最好的个体和最好的函数值
        # print("best_individual: "+ str(best_individual))
        v1, v2, v3 = b2d(best_individual)
        results.append([best_fit, v1, v2, v3]) #每次繁殖，将最好的结果记录下来
        # print(str(best_individual) + " " + str(best_fit))
        selection(pop, fit_value) #自然选择，淘汰掉一部分适应性低的个体
        crossover(pop, pc) #交叉繁殖
        mutation(pop, pc) #基因突变
    # print(results)
    results.sort()
    # wirte results to file
    writeToFile(results, "generation_" + str(generations) + ".txt")
    print(results[-1])

 
 
if __name__ == '__main__':
    gen = [20]
    for g in gen:
        generAlgo(int(g))



#df_test = pd.read_csv('test_set.csv') 
#df_test.drop(columns=['article'],inplace=True)

#x_test = vectorizer.transform(df_test['word_seg'])
#y_test = model.predict(x_test)  
      
#df_test['class'] = y_test.tolist()
#df_test['class'] = df_test['class']+1
#df_result = df_test.loc[:,['id','class']]
#df_result.to_csv('result8.csv',index=False)       
        

#模型存储
#from sklearn.externals import joblib


#joblib.dump(value=model,filename="LR_GAmodel.gz",compress=True)
#print("model has saved!!")

#模型调用
#model=joblib.load(filename="ridgeModel.gz")
#print(type(model))
#result2=model.predict(testSet)
#print(result2)        
        
