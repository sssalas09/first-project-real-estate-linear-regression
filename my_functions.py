# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 20:43:18 2021

@author: sssalas
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import r2_score
import math


def my_r2_plot(y_train, y_hat_train):
    """This create a plot with r-squared"""
    plt.figure() 
    plt.scatter(x=y_train,y=y_hat_train) 
    plt.plot(y_train,y_train,c="red") 
    r_squared = r2_score(y_train, y_hat_train)
    plt.title('R-squared equals %.3f' %r_squared) 



def train_test_alignment(train, test):
    """This aligns the train and test data"""
# Get missing columns in the training test
    missing_cols = set( train.columns ) - set( test.columns )
# Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        test[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
    test = test[train.columns]



def dict_zip(a,b):
    """This zips two sets of one-columned data type with one dimension into a dictionary"""
    names_coef = dict(zip(a,b))
    return names_coef


def len_unique_print(c,d):
    """This prints the length/number of unique values in a given dataset (c) and list (d)"""
    for i in d:
        print(len(pd.unique(c[i])))


def how_many_features(n,d):
	"""This returns the number of features we are expecting when we use polynomial features 
	given n features and d degrees"""
    return math.factorial(n+d)/(math.factorial(n+d-d)*math.factorial(d))

def print_shape(a):
    for i in a:
        print(i.shape)