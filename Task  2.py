#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("customer_booking.csv")
df.head()


# In[3]:


#Exploratory Data Analysis


# In[4]:


df.dtypes


# In[5]:


df.shape


# In[6]:


df.isnull().sum()


# In[7]:


sns.set(font_scale=1)

plt.title('Checking the Null Values')
sns.heatmap(df.isnull())


# In[8]:


grouped = df.groupby('booking_origin').size()
# Sort the resulting Series in ascending order by size
sorted_grouped = grouped.sort_values(ascending=False)
sorted_grouped.head(10)


# In[9]:


df.groupby('trip_type').size()


# In[10]:


df.groupby('sales_channel').size()


# In[11]:


##Machine Learning


# In[18]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn import metrics


# In[13]:


X = df.drop(['booking_complete'], axis=1)
y = df['booking_complete']

#changing object dtype to int dtype
for colname in X.select_dtypes("object"):
    X[colname], _ = X[colname].factorize()


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(y_test) # This is the output....


# In[15]:


model = RandomForestClassifier()


# In[16]:


model.fit(X_train,y_train)


# In[20]:


model.score(X_test,y_test)*100


# In[21]:


model.predict(X_test)


# In[22]:


y_predicted = model.predict(X_test)

cm = confusion_matrix(y_test, y_predicted)
cm


# In[25]:


# Calculate the AUC-ROC score
roc_auc = roc_auc_score(y_test, y_predicted)
print('AUC-ROC:', roc_auc)


# In[26]:


plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True)
plt.xlabel("Predicted")
plt.ylabel("Truth")


# In[ ]:




