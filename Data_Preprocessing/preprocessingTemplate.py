# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 15:23:39 2019

@author: Krinza Momin
"""


#Data Preprocessing
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing Dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3:].values


#splitting the dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

#feature Scaling (standardise/normalize)
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''
