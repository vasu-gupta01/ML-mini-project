# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 17:54:37 2022

@author: Punar Vasu Gupta
"""
from turtle import shape
import numpy as np

def train(X_train, y_train, X_val, y_val, MaxIter=50, 
alpha=0.001, batch_size=100, decay=0.0):
    N_train = X_train.shape[0]
    N_val = X_val.shape[0]

    # initialization
    w = np.zeros([X_train.shape[1], 1])
    # w: (d+1)x1

    losses_train = []
    risks_val = []
    acc_val = []

    w_best = None
    risk_best = 10000
    acc_best = 0
    epoch_best = 0

    for epoch in range(MaxIter):
        print('Epoch ' + str(epoch+1) + '/' + str(MaxIter) + '...')
        loss_this_epoch = 0
        for b in range(int(np.ceil(N_train/batch_size))):

            X_batch = X_train[b*batch_size: (b+1)*batch_size]
            y_batch = y_train[b*batch_size: (b+1)*batch_size]
    
            y_hat_batch, loss_batch, _ , _= predict(X_batch, w, y_batch)
            loss_this_epoch += loss_batch

            # TODO: Your code here
            # Mini-batch gradient descent
            gradient = (1/X_batch.shape[0])*(np.dot(X_batch.T, y_hat_batch - y_batch))
            w = w - (alpha * gradient)

        # TODO: Your code here
        # monitor model behavior after each epoch
        # 1. Compute the training loss by averaging loss_this_epoch
        loss_this_epoch = loss_this_epoch / (int(np.ceil(N_train/batch_size)))
        losses_train.append(loss_this_epoch)
        
        # 2. Perform validation on the validation set by the risk
        _, val_loss , risk, acc = predict(X_val, w, y_val)
        risks_val.append(risk)
        acc_val.append(acc)
        print('val_loss: '+ str(val_loss) + ', val_risk: '+ str(risk), 'val_acc: ' + str(acc))
        
        # 3. Keep track of the best validation epoch, risk, and the weights
        if acc > acc_best:
            acc_best = acc
            w_best = w
            epoch_best = epoch

    # Return some variables as needed
    return w_best, epoch_best, risk_best, risks_val, losses_train, acc_val



def predict(X, w, y=None):
    # X_new: Nsample x (d+1)
    # w: (d+1) x 1
    # y_new: Nsample

    # TODO: Your code here
    y_hat = np.dot(X, w)
    loss = 0.5 * np.mean(np.dot((y_hat - y).T, y_hat - y))
    risk = np.mean(abs(y_hat - y))
    y_hat = y_hat > 0.5
    acc = np.sum(y==y_hat)/np.size(y)

    return y_hat, loss, risk, acc