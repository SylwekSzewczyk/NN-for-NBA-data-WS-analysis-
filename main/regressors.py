# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 22:13:55 2019

@author: Sylwek Szewczyk
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from yellowbrick.regressor import PredictionError, ResidualsPlot
from data import df, df1
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
import numpy as np
#1 Data preprocessing

X = df.iloc[:,:-2].values
y = df.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#2 Build regressors

regressor = Lasso(alpha=0.005, random_state=0)
regressor.fit(X_train, y_train)

regressor1 = RandomForestRegressor(n_estimators=300, random_state=0)
regressor1.fit(X_train,y_train)
visualiser = PredictionError(regressor)
visualiser.fit(X_train, y_train)
visualiser.score(X_test, y_test)
visualiser.poof()

visualiser1 = ResidualsPlot(regressor1)
visualiser1.fit(X_train, y_train)
visualiser1.score(X_test, y_test)
visualiser1.poof()


y_pred1 = regressor1.predict(X_test)

importance = pd.Series(np.abs(regressor.coef_.ravel()))
importance.index = df.columns.values.tolist()[:20]
importance.sort_values(inplace=True, ascending=False)
importance.plot.bar()
plt.ylabel('Lasso Coefficients')
plt.title('Feature Importance')