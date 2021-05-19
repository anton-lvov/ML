# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 09:58:15 2021

@author: admin
"""

"""
simple regression
        
    y = b0 +b1*X1
    
Multiple linear refression

    y = b0+b1*x1 + b2*x2 + ... + bn*xn



Assumptions of a Linear Regression:
    1.Linearity
    2. Homoscedastinity
    3. Multivariate notmality
    4, Independence of errors
    5.Lack of multicollinearity
    
    
5 methods of building models:
    1.All-in:
        Prior Knolewge
        You have to
        Preparing for backward Elimination
        
    2.Backward Elimination:
        Select a significance level to stay in the model  (SL = 0.05)
        fit the full model withh all possible predictors
        Consider the predictor with the highest P-value.if P>SL go to step 4 otherwise go to FIN
        Remove the predictor
        Fit model without this variable (back to consider...)
        
    3.Forward selection:
        Select a significance level to enter the model  (SL = 0.05)
        Fit all simple refression models y~xn Select the one with the lowest P-value
        Keep this variable and fir all possible models with one extra predictor added to the one you already have
        Consider the predictor with the lowest P-value. If P< SL , got ot previous. otherwise FIN
        
    4.Bidirectional elimination:
         Select a significance level to enter the model and to stay in the model  (SLenter = 0.05, SLstay = 0.05)
         Perform next step of forward Selection  (P < SLenter to enter)
         Perform all steps of backward elimination ( P< SLstay to stay)
         No new variables can enter and no old variables can exit
         
        
    5.Score Comparison:
        
    6. All possible models:
        Select a criterion of goodness of fit ( Akaike criterion)
        Construct all possible regression models 2^N-1 total combinations
        Select the one with the best criterion
            
        
"""
#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv("50_Startups.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder',OneHotEncoder(),[3])],remainder = 'passthrough')
x = np.array(ct.fit_transform(x))

#splitting dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train, y_test = train_test_split(x,y,test_size=0.2, random_state = 0)

#Training the multiple Linear Regression model on the training test
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#Predict the Test set results
y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))














































