# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 21:02:24 2018

@author: IIST
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 19:24:46 2018
@author: IIST
"""
import numpy as np
import csv
from stochastic_gradient import stochastic_gradient_descent
from stochastic_gradient import find_rmse_gradient_descent
import time

def getdata(): 
    start=time.time()
    import pandas as pd
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    df = pd.read_csv(url, sep=';')
    result=df.values
    t_rows,t_cols=result.shape
    #Normalization
    for i in range(0,t_cols-1):
        mean=np.mean(result[:,i])
        std_dev=np.std(result[:,i])
        for j in range(0,t_rows):
            result[j,i]=(result[j,i]-mean)/std_dev
    hold_out1(result,t_rows,t_cols)
    end=time.time()
    print("The running time for online method is {}".format(end-start))
    
def hold_out1(result,rows,cols):
    np.random.shuffle(result) 
    train_rows=int(0.7*rows)  
    result_train=result[:train_rows]
    result_test=result[train_rows:rows]        
    hold_out(result_train,result_test)
    
def hold_out(train,test):
    w_list=np.zeros((train.shape[1],1), dtype=float)
    train_rows,cols=train.shape
    test_rows,cols=test.shape
    hold_out_iterations=2
    for hold in range(0,hold_out_iterations):
        np.random.shuffle(train)    
        X=np.delete(train,cols-1,1)
        y=np.delete(train, np.s_[0:cols-1], axis=1)
        X=np.insert(X,0,1,axis=1)                
        [alpha,rmse]=k_fold_function(train_rows,cols,X,y)
        w=stochastic_gradient_descent(X,y,train_rows,cols,alpha)
        w_list=np.add(w_list,w)
    w_list=w_list/hold_out_iterations
    X_test=np.delete(test,cols-1,1)
    y_test=np.delete(test, np.s_[0:cols-1], axis=1)
    X_test=np.insert(X_test,0,1,axis=1) 
    rmse=find_rmse_gradient_descent(X_test,y_test,w_list)
    np.savetxt("w_after_stochastic.csv", w_list, delimiter=",")
    print("Final RMSE for 30% data is {}".format(rmse))
    
def k_fold_function(t_rows,t_cols,X,y):          
    alpha_list=[1e-5]   
    min_rmse=50
    k_folds=10
    k_fold_test_rows=int(t_rows/k_folds)
    k_fold_train_rows=t_rows-k_fold_test_rows
    
    for alpha in alpha_list:
        #print(alpha)
        rmse_array=[]
        for k_fold in range (0,k_folds):    
            X_test=X[:k_fold_test_rows]
            y_test=y[:k_fold_test_rows]
            
            X_train=X[k_fold_test_rows:t_rows]
            y_train=y[k_fold_test_rows:t_rows]
            
            w=stochastic_gradient_descent(X_train,y_train,k_fold_train_rows,t_cols,alpha)
            rmse_array.append(find_rmse_gradient_descent(X_test,y_test,w))
            
            X_new=np.concatenate((X_train,X_test))
            y_new=np.concatenate((y_train,y_test))
            X=np.copy(X_new)
            y=np.copy(y_new)
        avg_rmse=np.average(rmse_array)
        if(avg_rmse<min_rmse):
            min_rmse=avg_rmse
            min_alpha=alpha
            #min_w=np.copy(w)
            #print(min_rmse)
    print("Alpha is {}, and min_rmse is {}".format(min_alpha,min_rmse))
    return (min_alpha,min_rmse)

getdata();