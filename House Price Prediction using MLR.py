# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 06:53:51 2022

@author: MANAS
"""

# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step - 1 : Importing the dataset
dataset = pd.read_csv(r'C:\Users\MANAS\Desktop\Mission DS\Workshop\kc_house_data.csv')
dataset.head() # 21 columns
print(dataset.dtypes) # Checking for categorical data, id & date are not useful,so eliminated

# Dropping the id & date column
dataset = dataset.drop(['id', 'date'], axis = 1)
dataset.head()  # 19 columns
print(dataset.dtypes)

# Here Price is the dependent variable, hence y = price, rest all attributes are x

# Step - 2 : Separating independent (x) & dependent(y) variables
x = dataset.iloc[:,1:] # All column except 1st - price
y = dataset.iloc[:,0] # 1st column- price

# Step - 3 : Splitting dataset into test & train datasets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

# Step - 4 : Model Building
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
# Predicting the Test set results
y_pred = regressor.predict(x_test)

# Step - 5 : Backward Elimination
# In x, we have 18 attributes, among it, which one we need to use, for that using this technique
import statsmodels.formula.api as sm
x=np.append(arr=np.ones((21613,1)).astype(int),values=x,axis=1)#Added column constant = 1 for all 21613 rows

# Step - 6 : Optimization
import statsmodels.api as sm
x_opt = x[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary() 

# Now testing individually all x-attributes p-value < 0.05
# found x4 & x5 p-values are > 0.05, hence eliminating these 2 columns now

 import statsmodels.api as sm
 x_opt = x[:,[0,1,2,3,6,7,8,9,10,11,12,13,14,15,16,17,18]]
 regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
 regressor_OLS.summary()
