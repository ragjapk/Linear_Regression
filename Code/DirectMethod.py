# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 13:48:56 2018

@author: Ragja
"""

import csv
import numpy as np
import math
import time
start = time.time()

#input data file

reader = csv.reader(open("Data3.csv", "rt"), delimiter=",")
x = list(reader)
result = np.array(x).astype("float")

np.random.shuffle(result)
print(result.shape)
t_rows,t_cols=result.shape

#Normalization of features

for i in range(0,t_cols):
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

print(y_train.shape)
print(y_test.shape)

#Least Square Method 
Xt=np.transpose(X_train)
XtX=np.dot(Xt,X_train)
inverse = np.linalg.inv(XtX)
Xty=np.dot(Xt,y_train)
w=np.dot(inverse,Xty)
sum=0.0
f_xi=np.dot(X_test,w)
for i in range(0,test_rows):
    sum=sum+((f_xi[i]-y_test[i])**2)
    
rmse=math.sqrt(sum/test_rows)
print(rmse)
#np.savetxt("w_direct_method.csv", w, delimiter=",")
end = time.time()
print("Running time is {}".format(end-start))