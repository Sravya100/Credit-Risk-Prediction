#!/usr/bin/env python
# coding: utf-8

# In[101]:


#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score


# In[102]:


#importing datasets
train = pd.read_csv('/Users/sravyavarma/Desktop/CS 584/HW2/train.csv')
train_x = train.iloc[:, :-1].values
train_y = train.iloc[:, -1].values
test = pd.read_csv('/Users/sravyavarma/Desktop/CS 584/HW2/test.csv')
test_x = test.iloc[:,:].values


# In[103]:


#Encoding variables
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [10,11])], remainder='passthrough')
train_x = np.array(ct.fit_transform(train_x))
test_x = np.array(ct.fit_transform(test_x))   


# In[104]:


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_x = sc.fit_transform(train_x)
test_x = sc.transform(test_x)


# In[105]:


#Random Forest 
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 300, criterion = 'gini')
classifier.fit(train_x, train_y)
pred_y = classifier.predict(test_x)
np.savetxt('rfc.txt',pred_y,fmt="%i")

