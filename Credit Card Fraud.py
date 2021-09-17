#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


# Loading the dataset to a pandas dataframe


# In[3]:


credit_card_data = pd.read_csv("C:\\Users\\Siddharth Khorwal\\Desktop\\Python\creditcard.csv")


# In[ ]:





# In[4]:


# First 5 rows of the dataset
credit_card_data.head()


# In[5]:


# Last 5 rows of the dataset
credit_card_data.tail()


# In[6]:


# Dataset information 
credit_card_data.info()


# In[7]:


# Checking number of missing values in each column of the dataset
credit_card_data.isnull().sum()


# In[8]:


# Distribution of normal and fraudulent transactions
credit_card_data.value_counts('Class')


# This dataset is highly imbalanced
# 
# 0 = Normal Transaction
# 
# 1 = Fraudulent Transaction

# In[9]:


# Now separating the data for analysis 
Normal = credit_card_data[credit_card_data.Class == 0]
Fraud = credit_card_data[credit_card_data.Class == 1]


# In[10]:


print(Normal.shape)
print(Fraud.shape)


# In[11]:


# statistical summary of the normal transactions
Normal.Amount.describe()


# In[12]:


# statistical summary of the fraudulent transactions
Fraud.Amount.describe()


# In[13]:


# Comparing the values of both the transactions
credit_card_data.groupby('Class').mean()


# To deal with the imbalanced data, we will use Under-Sampling
# 
# 

# In[14]:


# Building a sample dataset containing similar distribution of normal and fraudulent transaction
# Number of fraudulent transaction = 492


# In[15]:


Normal_sample = Normal.sample(n=492)


# In[16]:


# Concatenating the two Dataframes


# In[17]:


New_cc_data = pd.concat([Normal_sample, Fraud], axis=0)


# In[18]:


New_cc_data.head()


# In[19]:


New_cc_data.tail()


# In[20]:


New_cc_data.value_counts('Class')


# In[21]:


New_cc_data.groupby('Class').mean()


# In[22]:


# Now splitting the data into Features and Targets
X = New_cc_data.drop('Class', axis=1)
Y = New_cc_data['Class']


# In[23]:


print(X)


# In[24]:


print(Y)


# In[25]:


# Now splitting the data into Training and Testing set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# In[26]:


print(X.shape, x_train.shape, x_test.shape)


# In[27]:


# Model Training
# Logistic Regression Model Fitting


# In[28]:


model = LogisticRegression()


# In[29]:


# Training the Logistic Regression model with the Training Data


# In[36]:


model.fit(x_train, y_train)


# In[31]:


# Predicting the train set results and calculating the accuracy
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)


# In[32]:


print('Accuracy score on Training Data:', training_data_accuracy)


# In[33]:


# Accuracy on test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)


# In[34]:


print('Accuracy score on Test Data:', test_data_accuracy)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




