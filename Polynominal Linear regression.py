# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 10:04:36 2021

@author: admin
"""

"""
Simple Linear Regression  :  y = b0+b1X1
Multiple Linear regression: y= b0+b1x1+b2x2+ ....+bn+xn
Polynomial Linear Regression: y =b0+b1x1+b2x1^2+ ... + bnx1^n 
"""
#import libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset

dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#Training the Simple Linear regression model on the Whole set

from sklearn.linear_model import LinearRegression
Lin_reg = LinearRegression()
Lin_reg.fit(x,y)

#Training the Simple Linear regression model on the Whole set

from sklearn.preprocessing import PolynomialFeatures
Pol_reg = PolynomialFeatures(degree = 4)
X_poly = Pol_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)
"""
#visualising the Linear results
plt.scatter(x,y,color = 'red')
plt.plot(x,Lin_reg.predict(x),color = 'blue')
plt.title('Truth or bluff (Linear)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visualizing the Polynomial results
plt.scatter(x,y,color = 'red')
plt.plot(x,lin_reg_2.predict(X_poly),color = 'blue')
plt.title('Truth or bluff (Polynomial)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
"""

#Visualizing the Polynomial results higher resolution
X_grid = np.arange(min(x),max(x),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(x,y,color = 'red')
plt.plot(X_grid,lin_reg_2.predict(Pol_reg.fit_transform(X_grid)),color = 'blue')
plt.title('Truth or bluff (Polynomial)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predict a Linear result
print(Lin_reg.predict([[6.5]]))
# Predict Polynomial result
print(lin_reg_2.predict(Pol_reg.fit_transform([[6.5]])))
