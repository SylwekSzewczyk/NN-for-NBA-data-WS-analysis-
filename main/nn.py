# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 22:10:26 2019

@author: Sylwek Szewczyk
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data import df

#1 Data preprocessing

X = df.iloc[:,:-2].values
y = df.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
