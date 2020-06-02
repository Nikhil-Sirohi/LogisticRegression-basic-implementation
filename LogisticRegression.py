#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pt


# In[23]:


df=pd.read_csv('bank-additional.csv',delimiter=";")
df.head()


# In[24]:


df.shape


# #DATA PREPROCESSING

# In[25]:


y=df["y"]
x=df.drop(["y"],axis=1)


# In[26]:


from sklearn.preprocessing import LabelEncoder
LEC=LabelEncoder()
for i in range(len(x.columns)):
    x.iloc[:,i]=LEC.fit_transform(x.iloc[:,i])


# In[27]:


x.head()


# In[28]:


df.applymap(np.isreal).head()


# In[33]:


from sklearn.preprocessing import StandardScaler
stc=StandardScaler()
x=stc.fit_transform(x)


# In[32]:


x


# In[35]:


pd.DataFrame(x).head()


# In[37]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


# In[40]:


#TRAIN MODEL
from sklearn.linear_model import LogisticRegression
lre=LogisticRegression()
lre.fit(x_train,y_train)


# In[45]:


#TEST MODEL AND PREDICT
y_pred=lre.predict(x_test)


# In[48]:


#CHECKING ACCURACY OF MODEL
from sklearn import metrics
print("Accuracy",metrics.accuracy_score(y,lre.predict(x)))


# In[53]:


df3=pd.DataFrame({"Actual":y.values,"Predicted":lre.predict(x)})
df3.head(15)


# In[59]:


number_of_unequal_values=0
for i in range(len(df3)):
    if df3.iloc[i,0]!=df3.iloc[i,1]:
     number_of_unequal_values=number_of_unequal_values+1
  
print(number_of_unequal_values)


# In[ ]:




