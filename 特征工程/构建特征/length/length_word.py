'''
将原始数据的word的长度特征，并将结果保存到本地

article特征可做类似处理

'''
df_train=pd.read_csv('train_set.csv')
df_test=pd.read_csv('test_set.csv')


def get_word_len(df_series):
	word_len=[]
	for row in df_series:
    	word_len.append(len(row.split(' ')))
    return word_len

df_train_word = pd.DataFrame({'id':df_train['id'].values.tolist(),'word_len':get_word_len(df_train['word_seg'])})
df_test_word = pd.DataFrame({'id':df_test['id'].values.tolist(),'word_len':get_word_len(df_test['word_seg'])})


df_train_word.to_csv('./train_word_len.csv',index=False)
df_test_word.to_csv('./test_word_len.csv',index=False)


'''
df_train=pd.read_csv('train_set.csv')
df_test=pd.read_csv('test_set.csv')

train_word_len = []
for row in df_train['word_seg']:
    train_word_len.append(len(row.split(' ')))

test_word_len = []
for row in df_test['word_seg']:
    test_word_len.append(len(row.split(' ')))


df_train_word = pd.DataFrame(columns=['id','word_len'])
df_test_word = pd.DataFrame(columns=['id','word_len'])


df_train_word['id'] = df_train['id']
df_test_word['id'] = df_test['id']

df_train_word['word_len'] = train_word_len
df_test_word['word_len'] = test_word_len

df_train_word.to_csv('./train_word_len.csv',index=False)
df_test_word.to_csv('./test_word_len.csv',index=False)

'''

