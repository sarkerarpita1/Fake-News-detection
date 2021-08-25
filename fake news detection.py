#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


fake=pd.read_csv('Fake.csv')
fake.head()


# In[3]:


true=pd.read_csv('True.csv')
true.head()


# In[4]:


fake["lable"]=0
true["lable"]=1


# In[5]:


fake.shape


# In[6]:


fake.head()


# In[7]:


news=pd.concat([fake.iloc[:,:],true.iloc[:,:]],axis=0,ignore_index=True)


# In[8]:


news.head()
news.tail()


# In[9]:


news.shape


# In[10]:


news.subject.unique()


# In[11]:


news.isnull().sum()


# In[12]:


news.drop(["title","subject","date"],axis=1,inplace=True)


# In[13]:


news.head()


# In[14]:


news.text[4]


# In[15]:


import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score,confusion_matrix


# In[16]:


lables=news["lable"]


# In[17]:


x_train,x_test,y_train,y_test=train_test_split(news['text'],lables,test_size=0.2,random_state=7)


# In[18]:


x_train.head
x_train.shape


# In[19]:


y_test.head()
y_test.shape


# In[20]:


tf_vec=TfidfVectorizer(stop_words="english",max_df=0.7)


# In[21]:


tf_train=tf_vec.fit_transform(x_train)
tf_test=tf_vec.transform(x_test)


# In[22]:


tf_train.shape
tf_test.shape


# In[23]:


pac=PassiveAggressiveClassifier(max_iter=60)
#fiting model
pac.fit(tf_train,y_train)


# In[24]:


y_pred=pac.predict(tf_test)
y_pred


# In[25]:


score=accuracy_score(y_test,y_pred)
score


# In[26]:


print(f'Accuracy: {round(score*100,2)}%')


# In[ ]:




