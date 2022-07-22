# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 19:41:41 2022

@author: MANAS
"""

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the data
dataset = pd.read_csv(r"C:\Users\MANAS\Desktop\Mission DS\Shared Files\11th\kc_house_data.csv")
dataset.head()
dataset.describe()
dataset.info()

# Among the 21 columns here, we will consider only 2 columns
# x - sqft living
# y - price

x = dataset['sqft_living']
y = dataset['price']

# Now converting the series data to array using numpy
x = np.array(x).reshape(-1,1)
y = np.array(y)

# Splitting the data into Train & Test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=1/3, random_state=0)

# Fitting Simple Linear Regression into the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)  # Predicting the price

# Visualizing Training data
plt.scatter(x_train, y_train, color ='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Space Sq Ft Vs Price (Training Set)')
plt.xlabel('Sq Ft Space')
plt.ylabel('Price')
plt.show()

# Visualizing Testing data
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Space Sq Ft Vs Price (Testing Set)')
plt.xlabel('Sq Ft Space')
plt.ylabel('Price')
plt.show()