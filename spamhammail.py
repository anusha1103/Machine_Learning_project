#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


# In[4]:


# load the dataset to pandas Data Frame
raw_mail_data = pd.read_csv('spamham.csv')
# replace the null values with a null string
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')


# In[5]:


mail_data.shape


# In[6]:


mail_data.head() #sample data


# In[7]:


# label spam mail as 0; Non-spam mail (ham) mail as 1.
mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1


# In[8]:


# separate the data as text and label. X --> text; Y --> label
X = mail_data['Message']
Y = mail_data['Category']


# In[9]:


print(X)
print('.............')
print(Y)


# In[10]:


# split the data as train data and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=3)


# In[18]:


# transform the text data to feature vectors that can be used as input to the SVM model using TfidfVectorizer
# convert the text to lower case letters
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase='True')
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

#convert Y_train and Y_test values as integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


# In[19]:


# training the support vector machine model with training data
model = LinearSVC()
model.fit(X_train_features, Y_train) 


# In[20]:


# prediction on training data
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)


# In[21]:


print('Accuracy on training data : ', accuracy_on_training_data)


# In[22]:


# prediction on test data
prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)


# In[23]:


print('Accuracy on test data : ', accuracy_on_test_data)


# In[24]:


input_mail = ["I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times."]
# convert text to feature vectors
input_mail_features = feature_extraction.transform(input_mail)

#making prediction
prediction = model.predict(input_mail_features)
print(prediction)

if (prediction[0]==1):
  print('HAM MAIL')
else:
  print('SPAM MAIL')


# In[ ]:




