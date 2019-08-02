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
from sklearn.feature_selection import SelectFromModel
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

# Feature engineering
sel_ = SelectFromModel(Lasso(alpha=0.005, random_state=0)) # remember to set the seed, the random state in this function
sel_.fit(df.drop(['WShares/48'], axis=1), df['WShares/48'])
sel_.get_support

selected_feat = df.drop(['WShares/48'], axis=1).columns[(sel_.get_support())]

# let's print some stats
print('total features: {}'.format((X.shape[1])))
print('selected features: {}'.format(len(selected_feat)))
print('features with coefficients shrank to zero: {}'.format(
    np.sum(sel_.estimator_.coef_ == 0)))

# Model building
# Lasso
regressor = Lasso(alpha=0.005, random_state=0)
regressor.fit(X_train, y_train)
prediction_Lasso = regressor.predict(scaler.transform(np.array(values_topredict)))
# Random Forest Regressor
regressor1 = RandomForestRegressor(n_estimators=300, random_state=0)
regressor1.fit(X_train,y_train)
prediction_RFR = regressor1.predict(scaler.transform(np.array(values_topredict)))

visualiser = PredictionError(regressor)
visualiser.fit(X_train, y_train)
visualiser.score(X_test, y_test)
visualiser.poof()

visualiser1 = PredictionError(regressor1)
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