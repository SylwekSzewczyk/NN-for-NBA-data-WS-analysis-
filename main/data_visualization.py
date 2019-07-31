# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 22:04:38 2019

@author: Sylwek Szewczyk
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#2 Data checking

sns.heatmap(df.corr(), annot=True)
sns.distplot(df['WShares/48'])

sns.heatmap(df1.corr(), annot=True)
sns.distplot(df1['WShares/48'])
columns = df.columns.values.tolist()

def analyse_continous(df, var):
    df = df.copy()
    df[var].hist(bins=20)
    plt.figure()
    plt.ylabel('Number of players')
    plt.xlabel(var)
    plt.title(var)              
    plt.show()

for var in columns:
    analyse_continous(df, var)