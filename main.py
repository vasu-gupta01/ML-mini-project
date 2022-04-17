# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 15:58:46 2022

@author: Punar Vasu Gupta
"""
import numpy as np
import log_diabetes
import lin_diabetes
import matplotlib.pyplot as plt

def readData():
    data = np.genfromtxt('data/diabetes_binary_health_indicators_BRFSS2015.csv'
                         , delimiter=',', skip_header=1)
    
    random_data = np.arange(len(data))
    np.random.shuffle(random_data)
    data = data[random_data]
    
    train_data = data[:100000,:]
    random_train = np.arange(len(train_data))
    np.random.shuffle(random_train)
    train_data = train_data[random_train]
    X_train = train_data[:, 1:]
    X_train = np.concatenate(
        (np.ones([X_train.shape[0], 1]), X_train), axis=1)
    t_train = train_data[:, 1]
    t_train = t_train.reshape([len(t_train),1]) 
    
    val_data = data[100000:110000,:]
    random_val = np.arange(len(val_data))
    np.random.shuffle(random_val)
    val_data = val_data[random_val]
    X_val = val_data[:, 1:]
    X_val = np.concatenate(
        (np.ones([X_val.shape[0], 1]), X_val), axis=1)
    t_val = val_data[:, 1]
    t_val = t_val.reshape([len(t_val),1])
    
    
    test_data = data[110000:120000,:]
    random_test = np.arange(len(test_data))
    np.random.shuffle(random_test)
    test_data = test_data[random_test]
    X_test = test_data[:, 1:]
    X_test = np.concatenate(
        (np.ones([X_test.shape[0], 1]), X_test), axis=1)
    t_test = test_data[:, 1]
    t_test = t_test.reshape([len(t_test),1])
    
    return X_train, t_train, X_val, t_val, X_test, t_test
    

def get_accuracy(t, t_hat):
    """
    Calculate accuracy,
    """
    return np.sum(t==t_hat)/np.size(t)

def main():
    X_train, t_train, X_val, t_val, X_test, t_test = readData()
    
    print(X_train.shape, t_train.shape, X_val.shape, t_val.shape, X_test.shape,
          t_test.shape)
    
    MaxEpoch = 50
    
    # Linear Regression
    lin_w_best = lin_diabetes.train(X_train, t_train)
    t_hat = lin_diabetes.predict(X_test, lin_w_best)
    
    lin_accuracy = np.sum(t_test==t_hat)/np.size(t_test)
    print('Linear Regression Accuracy = ', lin_accuracy)
    
    
    epoch_best, acc_best,  W_best, train_losses, valid_accs = log_diabetes.train(
        X_train, t_train, X_val, t_val, 3)
    print('epoch_best = ', epoch_best)
    print('Validation Accuracy = ', acc_best)
    
    plt.figure()
    plt.plot(range(MaxEpoch), train_losses)
    plt.xlabel('epoch')
    plt.ylabel('Training Cross-Entropy Loss')
    plt.savefig('Training_Loss.png')

    plt.figure()
    plt.plot(range(MaxEpoch), valid_accs)
    plt.xlabel('epoch')
    plt.ylabel('Validation Accuracy')
    plt.savefig('Validation_Accuracy.png')
    
    # Test Performance
    print('Evaluating Test Performance...')
    y_hat, test_preds, _, acc_test = log_diabetes.predict(X_test, W_best, t_test)
    print('Test Accuracy = ', acc_test)
    

if __name__ == "__main__":
    main()