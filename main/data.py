# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 22:07:14 2019

@author: Sylwek Szewczyk
"""

import pandas as pd
import seaborn as sns

#1 Data cleaning

#a) 2018/2019 data
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

#b) 2004/2005 data

df1 = pd.read_html('https://www.basketball-reference.com/leagues/NBA_2005_advanced.html')[0]
df1.drop_duplicates(subset = 'Player', keep='first', inplace=True)
df1 = df1[df1['Player'] != 'Player']
df1.drop(['Unnamed: 19', 'Unnamed: 24', 'Pos', 'Age', 'Tm','Rk','Player'], axis = 1, inplace = True)
df1[df1.columns.values.tolist()] = df1[df1.columns.values.tolist()].astype(float)
df1[['G', 'MP']] = df1[['G', 'MP']].astype(int)
df1 = df1[df1['G'] >= 10]
df1.insert(22, 'WShares', df1['WS'], allow_duplicates = True)
df1.insert(23, 'WShares/48', df1['WS/48'], allow_duplicates = True)
df1.drop(['WS', 'WS/48'], axis = 1, inplace = True)