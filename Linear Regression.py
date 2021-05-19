# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 10:06:45 2021

@author: admin
"""


"""
y =b0+b1*X

Salary = bo + b1*experience

"""

#import libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset

dataset = pd.read_csv("Salary_Data.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


#splitting dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
x_train,x_test,y_train, y_test = train_test_split(x,y,test_size=0.2, random_state = 0)


#Training the Simple Linear regression model on the Training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


#predicting test set results
y_pred = regressor.predict(x_test)

#visualising the Training set results

plt.scatter(x_train,y_train,color = 'red')
plt.plot(x_train,regressor.predict(x_train),color = 'blue')
plt.title('Slary vs Experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

#Visualizing the Test results
plt.scatter(x_test,y_test,color = 'red')
plt.plot(x_train,regressor.predict(x_train),color = 'blue')
plt.title('Slary vs Experience (Test set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

#Simple prediction

print(regressor.predict([[12]]))  #[[12]] -> 12 scalar , [12] 1D, [[12]] 2D array.predict always expect 2d array
#138531,00 salary of employee with 12 years experience

#getting the final linear regression equation with the values of the coefficients

print(regressor.coef_) #9312,57
print(regressor.intercept_) #  26816.01

# Salary = 9312,57 * Years + 26816
