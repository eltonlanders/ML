# Data Preprocessing Tools

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X)

# Encoding categorical data
# Encoding the Independent Variable
#onehotencoder can be used with ordinal data, but results in more number 0f columns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)
# Encoding the Dependent Variable
#label encoder is not good when used with nominal data, but good with ordinal data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)   #a good test size is between 20% to 30%
print(X_train)
print(X_test)
print(y_train)
print(y_test)

# 2 main types standardization and normalization
# Scaling is necessary so that the models do not leave behind any feature when calculating the euclidean distance
# Scaling the dummy variables or not is to be decided
# for the dependent variable if it is regression then u have to do feature scaling and for binary values it is not needed
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])   #fit and transform for the train, to do on all just pass X_train
X_test[:, 3:] = sc.transform(X_test[:, 3:])   #only transform for the test
print(X_train)
print(X_test)