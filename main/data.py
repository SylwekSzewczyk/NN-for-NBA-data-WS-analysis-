# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 22:07:14 2019

@author: Sylwek Szewczyk
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#1 Data cleaning

df = pd.read_html('https://www.basketball-reference.com/leagues/NBA_2019_advanced.html')[0]
df.drop_duplicates(subset = 'Player', keep='first', inplace=True)
df = df[df['Player'] != 'Player']
df.drop(['Unnamed: 19', 'Unnamed: 24', 'Pos', 'Age', 'Tm','Rk','Player'], axis = 1, inplace = True)
df[df.columns.values.tolist()] = df[df.columns.values.tolist()].astype(float)
df[['G', 'MP']] = df[['G', 'MP']].astype(int)
df = df[df['G'] >= 10]
df.insert(22, 'WShares', df['WS'], allow_duplicates = True)
df.insert(23, 'WShares/48', df['WS/48'], allow_duplicates = True)
df.drop(['WS', 'WS/48'], axis = 1, inplace = True)

#2 Data preprocessing

X = df.iloc[:,:-2].values
y = df.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#3 Neural network implementation


