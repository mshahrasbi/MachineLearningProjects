
# Multiple Linear Regression

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data (state)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding dummy variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predicting the Test set results
y_pred = regressor.predict(X_test)

# building the optimal model using Backward elimination
import statsmodels.formula.api as sm 
# step 1
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
# step 2  Ordinary least Squares model (OLS)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()  
# step 3  
regressor_OLS.summary()
#
# remove X2 since it has P > SL highest value (0.990)
X_opt = X[:, [0, 1, 3, 4, 5]]
# step 2  Ordinary least Squares model (OLS)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()  
# step 3  
regressor_OLS.summary()
#
# remove X1 since it has P > SL highest value (0.990)
X_opt = X[:, [0, 3, 4, 5]]
# step 2  Ordinary least Squares model (OLS)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()  
# step 3  
regressor_OLS.summary()
#
# remove X4 since it has P > SL highest value (0.990)
X_opt = X[:, [0, 3, 5]]
# step 2  Ordinary least Squares model (OLS)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()  
# step 3  
regressor_OLS.summary()
#
# remove X5 since it has P > SL highest value (0.990)
X_opt = X[:, [0, 3]]
# step 2  Ordinary least Squares model (OLS)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()  
# step 3  
regressor_OLS.summary()