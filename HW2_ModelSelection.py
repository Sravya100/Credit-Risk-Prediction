#!/usr/bin/env python
# coding: utf-8

# In[84]:


#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score


# In[85]:


#Importing the dataset 
dataset = pd.read_csv('/Users/sravyavarma/Desktop/CS 584/HW2/train.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# In[86]:


#Encoding variables
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [10,11])], remainder='passthrough')
x = np.array(ct.fit_transform(x))


# In[87]:


#Train_test Split
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.25)


# In[88]:


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_x = sc.fit_transform(train_x)
test_x = sc.transform(test_x)


# In[91]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'gini')
classifier.fit(train_x, train_y)
pred_y = classifier.predict(test_x)
cm = confusion_matrix(test_y, pred_y)
print(cm)
f1_score(test_y, pred_y)


# In[93]:


#KNN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 100, metric = 'minkowski', p = 2)
classifier.fit(train_x, train_y)
pred_y = classifier.predict(test_x)
cm = confusion_matrix(test_y, pred_y)
print(cm)
f1_score(test_y, pred_y)


# In[94]:


#Linear SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear')
classifier.fit(train_x, train_y)
pred_y = classifier.predict(test_x)
cm = confusion_matrix(test_y, pred_y)
print(cm)
f1_score(test_y, pred_y)


# In[95]:


#Kernel SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf')
classifier.fit(train_x, train_y)
pred_y = classifier.predict(test_x)
cm = confusion_matrix(test_y, pred_y)
print(cm)
f1_score(test_y, pred_y)


# In[96]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(train_x, train_y)
pred_y = classifier.predict(test_x)
cm = confusion_matrix(test_y, pred_y)
print(cm)
f1_score(test_y, pred_y)


# In[97]:


#Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(train_x, train_y)
pred_y = classifier.predict(test_x)
cm = confusion_matrix(test_y, pred_y)
print(cm)
f1_score(test_y, pred_y)


# In[100]:


#Random Forest 
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 300, criterion = 'gini')
classifier.fit(train_x, train_y)
pred_y = classifier.predict(test_x)
cm = confusion_matrix(test_y, pred_y)
print(cm)
f1_score(test_y, pred_y)

