# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
#to get the target variable to the end
cols_at_end = ['Weight']
dataset = dataset[[c for c in df if c not in cols_at_end] 
        + [c for c in cols_at_end if c in df]]
"""

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))   

#Building the optimal model using backward selection

import statsmodels.api as sm
X=np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)   

X_opt=X[:, [0, 1, 2, 3, 4, 5]]
X_opt=np.array(X_opt, dtype=float)
regressor_OLS=sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())

X_opt=X[:, [0, 1, 2, 3, 4]]
X_opt=np.array(X_opt, dtype=float)
regressor_OLS=sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())

