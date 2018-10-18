#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 16:07:58 2018

@author: siyangzhang
"""

from sklearn.preprocessing import Imputer
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

df = pd.read_csv("/Users/siyangzhang/Desktop/课程/ie598 machine learning in finance/groupproject/MLF_GP1_CreditScore.csv")

df.isnull().any()

X, y = df.iloc[0:1700,0:26], df.InvGrd

print( X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=33)
print( X_train.shape, y_train.shape)

# Standardize the features

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#print summary of data frame
summary = X.describe()
print(summary)

