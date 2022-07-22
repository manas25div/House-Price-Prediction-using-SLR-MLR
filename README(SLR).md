# House-Price-Prediction-using-SLR-MLR
Data Pre-processing for House Price Prediction using SLR &amp; MLR

Import Libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

Read the data
dataset=pd.read_csv(r'C:\Users\MANAS\Desktop\Mission DS\Shared Files\11th\kc_house_data.csv') 
dataset.head() 
dataset.describe() 
dataset.info()

There are 21 columns here but for Linear Regression we used only 2 columns
X- sqft_living
y- Price
X= dataset['sqft_living'] 
y=dataset['price']

Now you have to convert this series data to Array using numpy
X=np.array(X).reshape(-1,1) 
y=np.array(y)

from sklearn.model_selection import train_test_split 
X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression 
regressor=LinearRegression() 
regressor.fit(X_train, y_train) 
y_pred=regressor.predict(X_test)

Visualizing Training Data
plt.scatter(X_train, y_train,color='red') 
plt.plot(X_train,regressor.predict(X_train),color='blue') 
plt.title('Space sq ft vs Price (Training set)') 
plt.xlabel('Sq ft space') 
plt.ylabel('Price') 
plt.show()

Visualizing Testing Data
plt.scatter(X_test, y_test,color='red') 
plt.title('Space sq ft vs Price (Testing data set)') 
plt.plot(X_train,regressor.predict(X_train),color='blue') 
plt.xlabel('Sq ft space') 
plt.ylabel('Price') 
plt.show()
