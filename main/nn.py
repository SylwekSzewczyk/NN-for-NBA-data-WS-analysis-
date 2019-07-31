# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 22:10:26 2019

@author: Sylwek Szewczyk
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data import df, df1
import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np

#1 Data preprocessing

X = df.iloc[:,:-2].values
y = df.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#2 NN architecture

nn = Sequential()
nn.add(Dense(units = 10, kernel_initializer = 'normal', activation = 'relu', input_dim = 20))
nn.add(Dense(output_dim=5, init = 'uniform', activation='relu'))
nn.add(Dense(output_dim=1, init = 'uniform', activation='linear'))
nn.compile(optimizer='adam', loss = 'mean_squared_error', metrics=['mae'])
nn.fit(X_train, y_train, batch_size = 50, epochs = 100)

y_pred = nn.predict(X_test)

topredict = df1.loc[df1['WShares']== max(df1['WShares'])]
topredict.drop(['WShares', 'WShares/48'], axis=1, inplace = True)
values_topredict = topredict.values.tolist()
prediction = nn.predict(scaler.transform(np.array(values_topredict)))

plt.scatter(y_pred, y_test)
plt.show()