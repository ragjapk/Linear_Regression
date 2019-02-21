# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 15:26:39 2018

@author: IIST
"""
import numpy as np
import math

def gradient_descent(X_train,y_train,train_rows,t_cols,alpha):    
    epsilon=0.00001
    w=np.zeros((t_cols,1), dtype=float)
    w_new=np.zeros((t_cols,1), dtype=float)
    w_difference=np.zeros((t_cols,1), dtype=float)
    f_xi=np.zeros((train_rows,1),dtype=float)    
    difference_vector=np.zeros((train_rows,1),dtype=float)   
    w_norm=10
    wt=np.transpose(w)
    while w_norm>epsilon:   
        for i in range(0,train_rows):    
            f_xi[i]=np.dot(wt,X_train[i])
            
        difference_vector=np.subtract(y_train,f_xi)    
        diff_transpose=np.transpose(difference_vector)
            
        for j in range(0,t_cols):
           k=np.dot(diff_transpose,X_train[:,j])
           w_new[j]=w[j]+ alpha*k
           
        w_difference=np.subtract(w_new,w)
        w=np.copy(w_new)
        wt=np.transpose(w)
        w_norm=np.linalg.norm(w_difference)
    return w

def find_rmse_gradient_descent(X_test,y_test,w):
    sum=0
    f_x=np.dot(X_test,w)
    for i in range(0,len(X_test)):
        sum=sum+((f_x[i]-y_test[i])**2)
        
    rmse=math.sqrt(sum/len(X_test))
    return rmse