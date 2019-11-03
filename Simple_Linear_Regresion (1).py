#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Simple Linear Regression
# Importing the libraries
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt   
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
#%matplotlib inline


# In[ ]:


# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
dataset #show the dataset


# In[ ]:


X = dataset.iloc[:, 0].values
Y = dataset.iloc[:, 1].values
plt.figure(figsize=(16, 8))
plt.scatter(X, Y)
plt.title('Salary vs Experience')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()


# In[ ]:


X = dataset['YearsExperience'].values.reshape(-1,1) #reshape (-1,1) means unknown row and one coloum
Y = dataset['Salary'].values.reshape(-1,1)


# In[ ]:


# split 80% of the data to the training set while 20% of the data to test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0) 


# In[ ]:


#training the algorithm and to import LinearRegresion() class
regressor = LinearRegression()  
regressor.fit(X_train, Y_train) # fit() method along with our training data


# In[ ]:


#To retrieve the intercept:
print(regressor.intercept_)

#For retrieving the slope:
print(regressor.coef_)


# In[ ]:


#Predicting the Test set result
Y_pred = regressor.predict(X_test)


# In[ ]:


dataset = pd.DataFrame({'Actual': Y_test.flatten(), 'Predicted': Y_pred.flatten()})
dataset


# In[ ]:


# Visualising the Training set results
plt.figure(figsize=(16, 8))
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience  (Training Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()


# In[ ]:


# Visualising the Test set results
plt.figure(figsize=(16, 8))
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_test, regressor.predict(X_test), color = 'blue')
plt.title('Salary vs Experience  (Test Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()


# In[ ]:


from sklearn import metrics
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred))


# In[ ]:


from sklearn.metrics import r2_score
score=r2_score(Y_test,Y_pred) #(predict-mean)/(actual-mean)
print(score)


# In[ ]:




