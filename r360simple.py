
# coding: utf-8

# In[58]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
user_info = pd.read_table('d:\\data\\user_info.txt',sep=',',header=0)
rong_tag = pd.read_table('d:\\data\\rong_tag.txt',sep=',',header=0)
relation1 = pd.read_table('d:\\data\\relation1.txt',sep=',',header=0)
relation2 = pd.read_table('d:\\data\\relation2.txt',sep=',',header=0)
consum = pd.read_table('d:\\data\\consumption_recode.txt',sep=',',header=0)
train =  pd.read_table('d:\\data\\train.txt',sep=',',header=0)
test = pd.read_table('d:\\data\\test.txt',sep=',',header=0)


# In[260]:

train_x = train['user_id']
train_y = train['lable']
test_x = test['user_id']


# In[184]:

#user_info = pd.read_table('/home/azureuser/pywork/data/user_info.txt',sep=',',header=0)
recent_user = user_info.fillna(-1)
recent_user=recent_user.replace('NONE','-1')
recent_user['age'] = recent_user.groupby(['user_id'])['age'].transform(max)
#recent_user['modifycount'] = recent_user.groupby(['user_id'])['user_id'].count()
#分组选出时间最近的用户
recent_user_1 = (recent_user.assign(temp=recent_user.groupby(['user_id'])['tm_encode'].rank(method='first', ascending=False)).query('temp<2').sort(['tm_encode','age'],ascending=False))
del recent_user_1['temp']

age_inx = recent_user_1[recent_user_1['age']=='-1']
recent_user_1[recent_user_1['age']=='-1']['age'] = recent_user_1['age'].mode()

#增加修改次数
tt = recent_user.groupby(['user_id'])['age'].count().to_frame('modifycount').reset_index()
recent_user_1 = pd.merge(recent_user_1,tt,on='user_id',how='left')

#增加rel2
r = relation2.groupby(['user1_id','user2_id']).count()
r = r.reset_index()
r = r.groupby(['user1_id'])['relation2_type'].sum().to_frame('rel2count').reset_index()
ttt = r.rename(columns ={'user1_id':'user_id'})
recent_user_1 = pd.merge(recent_user_1,ttt,on='user_id',how='left')

recent_user_1 = recent_user_1.fillna(0)

age_inx = recent_user_1[recent_user_1['age']=='-1']
recent_user_1[recent_user_1['age']=='-1']['age'] = recent_user_1['age'].mode()

train_x = train['user_id']
train_y = train['lable']
test_x = test['user_id']

train_input_x = train_x.to_frame('user_id')
test_input_x = test_x.to_frame('user_id')
train_input_x = pd.merge(train_input_x,recent_user_1,on='user_id',how='left')
test_input_x  = pd.merge(test_input_x,recent_user_1,on='user_id',how='left')
train = pd.merge(train,recent_user_1,on='user_id',how='left')


# In[120]:

recent_user_1.head(5)


# In[182]:

#r = relation2.groupby(['user1_id','user2_id']).count()
#r = r.reset_index()
#r = r.groupby(['user1_id'])['relation2_type'].sum().to_frame('rel2count').reset_index()
#r = r.rename(columns ={'user1_id':'user_id'})
#r



# In[185]:

rong_df = rong_tag.groupby(['user_id'])['rong_tag'].count().reset_index()
#rong_train = rong_df[rong_df['user_id'].isin(train_x)]
train_input_x.insert(23,'tags',0)
test_input_x.insert(23,'tags',0)
#recent_user.head(5)
train_input_x['tags'] = rong_df['rong_tag'].combine_first(train_input_x['tags'])
#tag_user = train_input_x.reset_index()
test_input_x['tags'] = rong_df['rong_tag'].combine_first(test_input_x['tags'])


# In[186]:

rr = relation1.groupby(['user1_id'])['user2_id'].count().reset_index()
train_input_x.insert(24,'rels',0)
test_input_x.insert(24,'rels',0)
train_input_x['rels'] = rr['user2_id'].combine_first(train_input_x['rels'])
#tag_user = train_input_x.reset_index()
test_input_x['rels'] = rr['user2_id'].combine_first(test_input_x['rels'])


# In[187]:

con = consum.groupby(['user_id'])['bill_id'].count().reset_index()
train_input_x.insert(25,'cons',0)
test_input_x.insert(25,'cons',0)
train_input_x['cons'] = con['bill_id'].combine_first(train_input_x['cons'])
#tag_user = train_input_x.reset_index()
test_input_x['cons'] = con['bill_id'].combine_first(test_input_x['cons'])


# In[98]:

#recent_user['age'] = recent_user.groupby(['user_id'])['age'].transform(max)


# In[188]:

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn import cross_validation

lr = LogisticRegression()


# In[190]:

array_feature = ['age','salary','education','marital_status','company_type','money_function','occupation','product_id','tm_encode','tags','rels','cons','modifycount']
train_input_x_1 = train_input_x[array_feature]
test_input_x_1 = test_input_x[array_feature]
#train_input_x_1 =train_input_x[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]]
#test_input_x_1 =test_input_x[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]]
X = StandardScaler().fit_transform(train_input_x_1)
test_X = StandardScaler().fit_transform(test_input_x_1)
lr.fit(X,train_y)
res = lr.predict_proba(test_X)
print(cross_validation.cross_val_score(lr,X,train_y,cv=5))


# In[ ]:




# In[253]:

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100,max_features='log2',min_weight_fraction_leaf=0.1)
model.fit(X,train_y)
#scores = cross_validation.cross_val_score(model, X, train_y, cv=3)
expected = train_y
predicted = model.predict(X)

#print(model.base_estimator_)

print(metrics.classification_report(expected, predicted))

#res = model.predict_proba(test_X)




# In[133]:

import numpy as np
result_p = res[:,1]
data = np.array(result_p)
df = pd.DataFrame(data)
test_x = pd.DataFrame(test_x)
df = df.rename(columns ={0:'probability'})
#dataframe命名
d = test_x.join(df)
d.to_csv('d:\\data\\test_with_random.txt',index=False,header=True)


# In[252]:

from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier()
model.fit(X,train_y)

expected = train_y
predicted = model.predict(X)

print(metrics.classification_report(expected, predicted))


# In[247]:

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X,train_y)

expected = train_y
predicted = model.predict(X)

print(metrics.classification_report(expected, predicted))


# In[246]:

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, train_y)

expected = train_y
predicted = model.predict(X)

print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

score = cross_validation.cross_val_score(model, X, train_y, cv=5)
score


# In[245]:

from sklearn import metrics
from sklearn.svm import SVC
# fit a SVM model to the data
model = SVC()
model.fit(X, train_y)
print(model)
# make predictions
expected = train_y
predicted = model.predict(X)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
score = cross_validation.cross_val_score(model, X, train_y, cv=5)
score


# In[233]:

tt = train_input_x
train_1 = train[train['lable']==1]
train_0 = train[train['lable']==0]


# In[234]:

ttt_1 = tt[tt['user_id'].isin(train_1['user_id'])]
ttt_0 = tt[tt['user_id'].isin(train_0['user_id'])]


# In[235]:

ttt_1[ttt_1['user_id']=='6c991fe85d9fcaea917f71fbdc9e384e']


# In[216]:

ttt_0.describe()


# In[254]:

user_info['age'].unique()


# In[283]:

train_user  = pd.merge(train,train_input_x,on='user_id',how='left')


# In[284]:

array_feature = ['lable','sex','age','salary','education','marital_status','company_type','money_function','occupation','product_id','tags','rels','cons','modifycount']
train_user = train_user[array_feature]


# In[285]:

train_user.corr()['lable']

