#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import numpy as np
import sklearn.datasets


# In[2]:


# getting the dataset
breast_cancer = sklearn.datasets.load_breast_cancer()


# In[3]:


print(breast_cancer)


# In[4]:


X = breast_cancer.data
Y = breast_cancer.target


# In[5]:


print(X)
print(Y)


# In[6]:


print(X.shape, Y.shape)


# In[7]:


import pandas as pd


# In[8]:


data = pd.DataFrame(breast_cancer.data, columns = breast_cancer.feature_names)


# In[9]:


data['class'] = breast_cancer.target


# In[10]:


data.head()


# In[11]:


data.describe()


# In[12]:


print(data['class'].value_counts())


# In[13]:


print(breast_cancer.target_names)


# In[14]:


data.groupby('class').mean()


# Train and Test Split

# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y)


# In[17]:


print(Y.shape, Y_train.shape, Y_test.shape)


# In[18]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
# test_size --> to specify the percentage of test data needed


# In[19]:


print(Y.shape, Y_train.shape, Y_test.shape)


# In[20]:


print(Y.mean(), Y_train.mean(), Y_test.mean())


# In[21]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y)
# stratify --> for correct distribution of data as of the original data 


# In[22]:


print(Y.mean(), Y_train.mean(), Y_test.mean())


# In[23]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)
# random_state --> specific split of data. each value of random_state splits the data differently


# In[24]:


print(X_train.mean(), X_test.mean(), X.mean())


# In[25]:


print(X_train)


#  Logistic Regression
# 

# In[26]:


# import Logistic Regression from sklearn
from sklearn.linear_model import LogisticRegression


# In[27]:


classifier = LogisticRegression() # loading the logistic regression model to the variable "classifier"


# In[28]:


# training the model on training data
classifier.fit(X_train, Y_train)


# Evaluation of the model

# In[29]:


# import accuracy_score 
from sklearn.metrics import accuracy_score


# In[30]:


prediction_on_training_data = classifier.predict(X_train)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)


# In[31]:


print('Accuracy on training data : ', accuracy_on_training_data)


# In[32]:


# prediction on test_data
prediction_on_test_data = classifier.predict(X_test)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)


# In[33]:


print('Accuracy on test data : ', accuracy_on_test_data)


# Detecting whether the Patient has breast cancer in benign or Malignant stage

# In[34]:


input_data = (13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259)
# change the input_data to numpy_array to make prediction
input_data_as_numpy_array = np.asarray(input_data)
print(input_data)

# reshape the array as we are predicting the output for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#prediction
prediction = classifier.predict(input_data_reshaped)
print(prediction) # returns a list with element [0] if Malignant; returns a listwith element[1], if benign.

if (prediction[0]==0):
  print('The breast Cancer is Malignant')
else:
  print('The breast cancer is Benign')


# In[ ]:




