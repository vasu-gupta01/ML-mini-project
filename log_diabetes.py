# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 17:54:08 2022

@author: Punar Vasu Gupta
"""
import numpy as np


def softmax(z):
    y = np.zeros(z.shape)
    
    for n in range(z.shape[0]):
        z_tmp = np.exp(z[n] - max(z[n]))
        y[n] = z_tmp / np.sum(z_tmp)
    
    return y


def predict(X, W, t=None):
    # X_new: Nsample x (d+1)
    # W: (d+1) x K

    # y : N x K    
    z = np.dot(X, W)
    y = softmax(z)   
    
    # t_hat : N x K
    t_hat = np.zeros(y.shape)
    for i in range(y.shape[0]):
        # get index of maximum value in y[i] sample.
        tmp = y[i,:].argmax(axis=0)
        t_hat[i, tmp] = 1

    y_tmp = np.where(y > 1e-16, y, 1e-16)
    log_y = np.where(y_tmp > 1e-16, np.log(y_tmp), 1e-16)
    
    loss = -np.mean(np.sum(log_y * t_hat))
    
    acc = 0
    if t is not None:
        cmp = 0
        for i in range(y.shape[0]):
            y_tmp = y[i].argmax(axis=0)
            cmp += (y_tmp == t[i])
        acc = float(cmp)
        acc /= len(t)
        
    return y, t_hat, loss, acc


def train(X_train, y_train, X_val, t_val, N_class, batch_size=100, alpha=0.1, 
          MaxEpoch=10, decay=0.):
    N_train = X_train.shape[0]
    N_val = X_val.shape[0]
    
    W = np.zeros([X_train.shape[1], N_class])
    
    train_losses = []
    valid_accs = []
    
    W_best = None
    epoch_best = 0
    acc_best = 0
    
    b_arr = np.arange(int(np.ceil(N_train/batch_size)))
    for epoch in range(MaxEpoch):
        print('Epoch ' + str(epoch+1) + '/' + str(MaxEpoch))
        loss_this_epoch = 0
        # np.random.shuffle(b_arr) # Testing meaningful hypothesis here.
        for b in b_arr:
            X_batch = X_train[b*batch_size: (b+1)*batch_size]
            y_batch = y_train[b*batch_size: (b+1)*batch_size]
            
            
            y_hat_batch, t_hat_batch, loss_batch, acc_batch = predict(X_batch, 
                                                                      W, y_batch)
            
            loss_this_epoch += loss_batch
            
            # one-hot encoding
            t_hat = np.zeros([y_batch.shape[0], N_class])
            for i in range(y_batch.shape[0]):
                t_hat[i, int(y_batch[i])] = 1
            
            gradient = (1/batch_size)*np.dot(X_batch.T, y_hat_batch - t_hat)
            
            l2 = gradient + decay*W
            W = W - (alpha * gradient)
        
        
        loss_this_epoch = loss_this_epoch / (int(np.ceil(N_train/batch_size)))
        train_losses.append(loss_this_epoch)
        
        _, _, val_loss, acc = predict(X_val, W, t_val)
        valid_accs.append(acc)
        print('val_loss: '+ str(val_loss) + ', val_accuracy: '+ str(acc))
        
        if acc > acc_best:
            acc_best = acc
            W_best = W
            epoch_best = epoch
        
    print('Done.')
    return epoch_best, acc_best,  W_best, train_losses, valid_accs