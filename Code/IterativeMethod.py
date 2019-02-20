# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 14:44:12 2018

@author: Ragja
"""

import csv
import numpy as np
import math

reader = csv.reader(open("Data3.csv", "rt"), delimiter=",")
x = list(reader)
result = np.array(x).astype("float")
#np.random.shuffle(result)
print(result.shape)
t_rows,t_cols=result.shape

#Normalization

for i in range(0,t_cols-1):
    mean=np.mean(result[:,i])
    std_dev=np.std(result[:,i])
    print(mean)
    print(std_dev)
    for j in range(0,t_rows):
        result[j,i]=(result[j,i]-mean)/std_dev



X=np.delete(result,t_cols-1,1)
y=np.delete(result, np.s_[0:t_cols-1], axis=1)
X=np.insert(X,0,1,axis=1)

train_rows=int(0.7*t_rows)
test_rows=t_rows-train_rows

X_train=X[:train_rows]
X_test=X[train_rows:t_rows]

y_train=y[:train_rows]
y_test=y[train_rows:t_rows]

print(X_train.shape)
print(y_test.shape)


#Intializing values for Gradient Descent

w=np.zeros((t_cols,1), dtype=float)
w_new=np.zeros((t_cols,1), dtype=float)
w_difference=np.zeros((t_cols,1), dtype=float)
f_xi=np.zeros((train_rows,1),dtype=float)
difference_vector=np.zeros((train_rows,1),dtype=float)

#Gradient Descent
alpha=0.001
epsilon=0.00001
w_norm=1
sum1=0
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
    print(w_norm)   
    
f_x=np.dot(X_test,w)
for i in range(0,test_rows):
    sum1=sum1+((f_x[i]-y_test[i])**2)
    
rmse=math.sqrt(sum1/test_rows)
print(rmse)

#np.savetxt("w_iterative_method.csv", w, delimiter=",")
       

    
    