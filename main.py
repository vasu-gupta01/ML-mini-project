# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 15:58:46 2022

@author: Punar Vasu Gupta
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import log_diabetes
import lin_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
    

def main():
    # Please download dataset into 'data\' from here: 
    # https://drive.google.com/file/d/1poNImi3SGcIX0FQs_0-8qnBXxA2LDWfY/view?usp=sharing    
    df = pd.read_csv('data/creditcard.csv')
    df = df.copy().drop('Time', axis=1)
    features = df.iloc[:, :-1]
    labels = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, 
                    stratify=labels, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, 
            y_train, test_size=0.25, random_state=0)

    MaxEpoch = 10
    
    # Conversions & Normalizing
    nX_train = X_train.to_numpy()
    nX_train = (nX_train - np.mean(nX_train, axis=0)) / np.std(nX_train, axis=0)
    ny_train = y_train.to_numpy()
    ny_train = ny_train.reshape([len(ny_train),1])
    
    nX_val = X_val.to_numpy()
    nX_val = (nX_val - np.mean(nX_val, axis=0)) / np.std(nX_val, axis=0)
    ny_val = y_val.to_numpy()
    ny_val = ny_val.reshape([len(ny_val),1])
    
    nX_test = X_test.to_numpy()
    nX_test = (nX_test - np.mean(nX_test, axis=0)) / np.std(nX_test, axis=0)
    ny_test = y_test.to_numpy()
    ny_test = ny_test.reshape([len(ny_test),1])
    
    print(nX_train.shape, ny_train.shape, nX_val.shape, ny_val.shape, nX_test.shape,
          ny_test.shape)
    
    # Linear Regression
    print('Training Linear Regression...')
    lin_w_best, lin_epoch_best, _, _, _ , val_acc= lin_diabetes.train(
        nX_train, ny_train, 
        nX_val, ny_val)
    print('Done.')
    
    # Test Performance
    print('Linear Regression epoch_best = ', lin_epoch_best)
    t_hat, _, lin_risk, lin_accuracy = lin_diabetes.predict(
        nX_test, lin_w_best, ny_test)
    print('Linear Regression Risk = ', lin_risk)
    print('Linear Regression Accuracy = ', lin_accuracy)
    # plot results
    plt.figure()
    plt.plot(range(50), val_acc)
    plt.xlabel('epoch')
    plt.ylabel('Validation Accuracy')
    plt.savefig('LinValidation_Accuracy.png')
    
    t_hat = t_hat > 0.5
    mat = confusion_matrix(y_test, t_hat)   # construct confusion matrix
    labels = ['Legitimate', 'Fraudulent']
    
    # plot confusion matrix as heat map
    sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False, cmap='OrRd',
                xticklabels=labels, yticklabels=labels)
    
    plt.xlabel('Predicted label')
    plt.ylabel('Actual label')
    plt.savefig('LinConsfusion.png')
    
    
    # Softmax Regression
    print('Training Softmax Regression...')
    epoch_best, acc_best,  W_best, train_losses, valid_accs = log_diabetes.train(
        nX_train, ny_train, 
        nX_val, ny_val, 2)
    print('Done.')
    print('epoch_best = ', epoch_best)
    print('Validation Accuracy = ', acc_best)
    
    # Plot Results
    plt.figure()
    plt.plot(range(MaxEpoch), valid_accs)
    plt.xlabel('epoch')
    plt.ylabel('Validation Accuracy')
    plt.savefig('Validation_Accuracy.png')
    
    # Test Performance
    print('Evaluating Test Performance...')
    y_hat, test_preds, _, acc_test = log_diabetes.predict(nX_test, W_best, ny_test)
    print('Test Accuracy = ', acc_test)
    
    test_preds = test_preds.argmax(axis=1)     # 1 = fraud, 0 = legitimate
    mat = confusion_matrix(y_test, test_preds)   # construct confusion matrix
    labels = ['Legitimate', 'Fraudulent']
    
    # plot confusion matrix as heat map
    sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False, cmap='OrRd',
                xticklabels=labels, yticklabels=labels)
    
    plt.xlabel('Predicted label')
    plt.ylabel('Actual label')
    plt.savefig('SoftmConsfusion.png')
    
    # Neural Network
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics="accuracy"
    )

    history = model.fit(X_train, y_train, epochs=10,
                validation_data=(X_test, y_test), batch_size=100)
    
    nn_acc = history.history['val_accuracy']
    # Plot Results
    plt.figure()
    plt.plot(range(MaxEpoch), nn_acc)
    plt.xlabel('epoch')
    plt.ylabel('Validation Accuracy')
    plt.savefig('NNValidation_Accuracy.png')
    
    # Based on the example by jeffprosise, we construct a confusion matrix due to the highly unbalanced data:
    # https://github.com/jeffprosise/Deep-Learning/blob/master/Fraud%20Detection.ipynb
    y_hat = model.predict(X_test) > 0.5     # 1 = fraud, 0 = legitimate
    mat = confusion_matrix(y_test, y_hat)   # construct confusion matrix
    labels = ['Legitimate', 'Fraudulent']
    
    # plot confusion matrix as heat map
    sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False, cmap='OrRd',
                xticklabels=labels, yticklabels=labels)
    
    plt.xlabel('Predicted label')
    plt.ylabel('Actual label')
    plt.savefig('NNConsfusion.png')

if __name__ == "__main__":
    main()