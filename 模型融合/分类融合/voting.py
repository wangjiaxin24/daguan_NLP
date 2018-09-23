import os
from collections import Counter

def read_dir_by_filter(root_dir,filter):
    file_list = []
    for root,dirs,files in os.walk(root_dir):
        for filepath in files:
            if(filter in filepath):
                file_list.append(os.path.join(root,filepath))
    return file_list

file_l = read_dir_by_filter('.','.csv')
# 10个文件

import pandas as pd 

res_l = []

for f in file_l:
	a = pd.read_csv(f)
	res_l.append(a['class'].values)

print(res_l)

# 把每个数组按照列combine

a_l = []
for i in range(len(res_l[0])):
	a_l.append([res_l[j][i] for j in range(10)])

def voting(class_l):
	final_class = []
	c_l = []
	for row in class_l:
		c = Counter(row)
		c_v_set = set(c.values())
		# 票数不等取最大
		if(len(c_v_set) > 1):
			res = max(c,key=c.get) 
		else: # 票数相等取最后一列的值
			res = row[-1]
		final_class.append(res)
		c_l.append(c)	
	return final_class,c_l

final_class,c = voting(a_l)


res_df = pd.DataFrame(columns=['id','class','counter'])

res_df['id'] = list(range(len(final_class)))
res_df['class'] = final_class
res_df['counter'] = c

res_df.to_csv('final_voting.csv',index=None)
	

