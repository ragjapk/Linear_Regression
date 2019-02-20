# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 18:27:38 2018

@author: IIST
"""

import csv
import numpy as np
import math
reader = csv.reader(open("Data4.csv", "rt"), delimiter=",")
x = list(reader)
result = np.array(x).astype("float")

np.random.shuffle(result)
print(result.shape)
t_rows,t_cols=result.shape

#Normalization via Z-Score
for i in range(0,t_cols):
    mean=np.mean(result[:,i])
    std_dev=np.std(result[:,i])
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

print(y_train.shape)
print(y_test.shape)

#Least square solution when N<n
Xt=np.transpose(X_train, axes=None)
XXt=np.dot(X_train,Xt)
XXt_inverse = np.linalg.inv(XXt)
XXt_inverse_y=np.dot(XXt_inverse,y_train)
w=np.dot(Xt,XXt_inverse_y)
sum=0.0
f_xi=np.dot(X_test,w)
for i in range(0,test_rows):
    sum=sum+((f_xi[i]-y_test[i])**2)
    
rmse=math.sqrt(sum/test_rows)
print(rmse)
np.savetxt("w_for_lagrange.csv", w, delimiter=",")
