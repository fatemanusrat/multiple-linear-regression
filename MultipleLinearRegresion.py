#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Multiple Linear Regression
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd
import seaborn as sns #for matrix
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
dataset


# In[ ]:


X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 4]


# In[ ]:


#data visualisation
sns.heatmap(dataset.corr()) #Plot rectangular data as a color-encoded for correlation matrix.


# In[ ]:


# categorical variables converted to numbers
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder() #integer-encoding of categorical column using the LabelEncoder
datasetlabel_encoder = dataset
datasetlabel_encoder.State = label_encoder.fit_transform(datasetlabel_encoder.State)
datasetlabel_encoder


# In[ ]:


X = datasetlabel_encoder[['State','Profit']].values
X


# In[ ]:


y = datasetlabel_encoder.Profit.values
y


# In[ ]:


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[0])


# In[ ]:


X = ohe.fit_transform(X).toarray() #fit_transform methods require array objects with shape (m, n) to be passed
X


# In[ ]:


X = X[:,1:]
X


# In[ ]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[ ]:


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[ ]:


# Predicting the Test set results
y_pred = regressor.predict(X_test)
print(y_pred)


# In[ ]:


#To retrieve the intercept:
print(regressor.intercept_)


# In[ ]:


#For retrieving the slope:
print(regressor.coef_)


# In[ ]:


from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred) #predict-mean/actual-mean
print(score)


# In[ ]:




