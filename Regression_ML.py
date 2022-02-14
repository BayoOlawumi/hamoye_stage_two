#!/usr/bin/env python
# coding: utf-8

# In[68]:


# Importing Necessary Libraries
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns


# In[69]:


# Data Import
df = pd.read_csv("energydata.csv")


# In[70]:


df_schema = pd.read_csv('schema.csv', index_col=None)
df_schema.head()


# ### Dataset Overview

# In[71]:


df.head()


# In[72]:


df.info()


# In[73]:


df.columns


# ## Question 12

# ### From the dataset, fit a linear model on the relationship between the temperature in the living room in Celsius (x = T2) and the temperature outside the building (y = T6). What is the R^2 value in two d.p.?

# In[86]:


from sklearn import metrics 
import sklearn.linear_model as linear_model


# In[75]:


df.head()


# In[179]:


# SELECT X AND Y FROM DATASET and Reshape to 2D
X = df['T2'].values.reshape(-1, 1)
Y = df['T6'].values.reshape(-1, 1)


# In[ ]:





# In[180]:


# Split data into Training and Testing Set
from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(X,Y,test_size=0.3, random_state=42)


# In[181]:


lnr = linear_model.LinearRegression()
lnr.fit(x_train, y_train)


# In[182]:


### Make Prediciton using Test Data
y_predicted = lnr.predict(x_test)


# In[183]:


r2_sc = metrics.r2_score(y_test, y_predicted)
r2_sc.round(2)


# ### Question 13

# ### Normalize the dataset using the MinMaxScaler after removing the following columns: [“date”, “lights”]. The target variable is “Appliances”. Use a 70-30 train-test set split with a random state of 42 (for reproducibility). Run a multiple linear regression using the training set and evaluate your model on the test set. Answer the following questions:
# 
# What is the Mean Absolute Error (in two decimal places)?

# In[99]:


# Drop date and Lights
df_new = df.drop(['date', 'lights'], axis=1)


# In[100]:


df_new.head()


# In[190]:


# Data Nomalization
from sklearn.preprocessing import MinMaxScaler
mx = MinMaxScaler()
scaled = mx.fit_transform(df_new)
normalized_df = pd.DataFrame(scaled, columns=df_new.columns)
x_normalized = normalized_df.drop(['Appliances'], axis=1)
y_normalized = normalized_df['Appliances']


# In[191]:


# Split Data in 70:30
x_train,x_test, y_train, y_test = train_test_split(x_normalized,y_normalized,test_size=0.3, random_state=42)


# In[192]:


# Train Model
lnr = linear_model.LinearRegression()
lnr.fit(x_train, y_train)


# In[193]:


y_predicted = lnr.predict(x_test)
y_predicted


# ### Question 13 - Mean Absolute Error

# In[194]:


mae = metrics.mean_absolute_error(y_test, y_predicted)
print("The Mean Absolute Error is ", mae.round(2))


# ### Question 14 - Residual Sum of Squares

# In[195]:


r_sum_square = ((y_test - y_predicted)** 2).sum().round(2)
print("The Residual Sum of Squares is ", r_sum_square)


# ### Question 15 - What is the Root Mean Squared Error (in three decimal places)?

# In[196]:


rmse = metrics.mean_squared_error(y_test, y_predicted)
print("The Root Mean Squared Error is ", rmse.round(3))


# ### Question 16 - What is the Coefficient of Determination (in two decimal places)?

# In[197]:


# Coefficient of determination also called as R2 score
r2_sc = metrics.r2_score(y_test, y_predicted)
print("The Root Mean Squared Error is ", r2_sc.round(2))


# ### Question 17 - Obtain the feature weights from your linear model above. Which features have the lowest and highest weights respectively?

# In[198]:


x_normalized.columns


# In[199]:


len(lnr.coef_)


# In[200]:


lowest = lnr.coef_.min()
highest = lnr.coef_.max()
print("The Lowest Weight", lowest)
print("The Lowest Weight", highest)


# In[202]:


# get importance
importance = lnr.coef_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0s, Score: %.5f' % (x_normalized.columns[i],v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.xlabel('Features Count')
plt.ylabel('Weight')
plt.grid('on')
plt.show()


# ### Question 18 - Train a ridge regression model with an alpha value of 0.4. Is there any change to the root mean squared error (RMSE) when evaluated on the test set?

# In[156]:


from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=0.4)
ridge_reg.fit(x_train, y_train)


# In[157]:


reg_y_predicted = ridge_reg.predict(x_test)
reg_y_predicted 


# In[158]:


reg_rmse = metrics.mean_squared_error(y_test, reg_y_predicted)
print("The Root Mean Squared Error is ", reg_rmse.round(3))


# ### Question 19 - Train a lasso regression model with an alpha value of 0.001 and obtain the new feature weights with it. How many of the features have non-zero feature weights?

# In[167]:


from sklearn.linear_model import Lasso
lasso_reg = Ridge(alpha=0.001)
lasso_reg.fit(x_train, y_train)


# In[187]:


# get importance
lasso_reg_importance = lasso_reg.coef_
# summarize feature importance
for i,v in enumerate(lasso_reg_importance):
	print('Feature: %0s, Score: %.5f' % (x_normalized.columns[i],v))
# plot feature importance
plt.bar([x for x in range(len(lasso_reg_importance))], lasso_reg_importance)
plt.xlabel('Features Count')
plt.ylabel('Weight')
plt.grid('on')
plt.show()


# In[171]:


len(lasso_reg.coef_)


# ### Question 20 - What is the new RMSE with the lasso regression? (Answer should be in three (3) decimal places)

# In[173]:


lasso_reg_y_predicted = lasso_reg.predict(x_test)
lasso_reg_y_predicted


# In[174]:


lasso_rmse = metrics.mean_squared_error(y_test, lasso_reg_y_predicted)
print("The Root Mean Squared Error is ", lasso_rmse.round(3))


# In[175]:


lasso_rmse


# In[176]:


reg_rmse


# In[178]:


rmse


# In[ ]:




