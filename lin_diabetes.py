# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 17:54:37 2022

@author: Punar Vasu Gupta
"""
import numpy as np

def train(X, t):
    """
    Given data, train your linear regression classifier.
    Return weight and bias
    """
    epochs = 1000
    alpha = 0.1
    
    # X_train = np.concatenate((np.ones([X.shape[0], 1]), X), axis=1)
    w = np.zeros([X.shape[1], 1])
    
    for i in range(epochs):
        y_hat = np.dot(X, w)
        gradient = (1/X.shape[0])*(np.dot(X.T, (y_hat - t)))
        
        w = w - (alpha * gradient)
        
    return w


def predict(X, w):
    """
    Generate predictions by your logistic classifier.
    """
    model = np.dot(X, w)
    
    t = np.zeros([model.shape[0], 1])
    
    for i in range(model.shape[0]):
        if(model[i][0] >= 0.5):
            t[i] = 1
        else:
            t[i] = 0
    
    return t