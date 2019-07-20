# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 13:46:28 2018

@author: IIST
"""
import csv 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import matplotlib.pyplot as plt
import math
class Classifier:
    dataset = np.array(list(csv.reader(open("crabs2.csv", "rt"), delimiter=","))).astype('float')
    
    def preprocess(self,output_index):
        X=np.delete(self.dataset,output_index,axis=1)
        y=np.take(self.dataset, output_index, axis=1)
        return X,y    
    
    def train_test_splitter(self):        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.3, stratify=self.y,)
        return X_train, X_test, y_train, y_test            

    def gradient_descent(self,X_train,y_train,train_rows,t_cols,alpha):    
        epsilon=0.001
        y_train=y_train.reshape((y_train.shape[0],1))
        w=np.zeros((t_cols,1), dtype=float)
        w_new=np.zeros((t_cols,1), dtype=float)
        w_difference=np.zeros((t_cols,1), dtype=float)
        f_xi=np.zeros((train_rows,1),dtype=float)    
        difference_vector=np.zeros((train_rows,1),dtype=float)   
        w_norm=10
        wt=np.transpose(w)
        steps=0
        while w_norm>epsilon and steps<100:   
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
            steps=steps+1
        return w    
        
    def gradient_ascent(self,X_train,y_train,train_rows,t_cols,alpha): 
        #print(train_rows,t_cols)
        #print(y_train.shape)
        epsilon=0.001
        y_train=y_train.reshape((y_train.shape[0],1))
        w=np.zeros((t_cols,1), dtype=float)
        w_new=np.zeros((t_cols,1), dtype=float)
        w_difference=np.zeros((t_cols,1), dtype=float)
        f_xi=np.zeros((train_rows,1),dtype=float)  
        temp=np.zeros((train_rows,1),dtype=float)  
        difference_vector=np.zeros((train_rows,1),dtype=float)   
        w_norm=10
        wt=np.transpose(w)
        while w_norm>epsilon:   
            for i in range(0,train_rows):    
                temp[i]=np.dot(wt,X_train[i])            #
            f_xi=1/(1+np.exp(-temp))
            difference_vector=np.subtract(y_train,f_xi)
            for j in range(0,t_cols):
               xx=X_train[:,j]
               k=np.dot(difference_vector.T,xx)
               w_new[j]=w[j]+ alpha*k
               
            w_difference=np.subtract(w_new,w)
            w=np.copy(w_new)
            wt=np.transpose(w)
            w_norm=np.linalg.norm(w_difference)
            #print(w_norm)
        #print(w)
        return w
    
    def prediction_function(self,X_test,y_test,w):
        temp=np.zeros((X_test.shape[0],1),dtype=float)  
        y_pred=np.zeros((y_test.shape[0],1),dtype=float)  
        for i in range(X_test.shape[0]):
                    temp[i]=np.dot(w.T,X_test[i])
                    if temp[i]>0:
                        y_pred[i]=1
                    else:
                        y_pred[i]=0
        return y_pred
    
    def performance_measures(self,y_test,y_pred):
        Tp,Fp,Fn,Tn=0,0,0,0
        for i in range(len(y_test)):
            if y_test[i]==1 and y_pred[i]==1:
                Tp=Tp+1
            elif y_test[i]==0 and y_pred[i]==0:
                Tn=Tn+1
            elif y_test[i]==1 and y_pred[i]==0:
                Fn=Fn+1
            else:
                Fp=Fp+1
        return Tp,Fp,Fn,Tn
    
    def get_final_performance(self,tp,fp,tn,fn):
        sensitivity=tp/(tp+fn)
        specificity=tn/(tn+fp)
        precision=tp/(tp+fp)
        accuracy=(tp+tn)/(tp+tn+fp+fn)
        f_measure=(2*precision*sensitivity)/(sensitivity+precision)
        return sensitivity,specificity,precision,accuracy,f_measure
    
    def cross_validator(self,hyper_list,function):
        skf = StratifiedKFold(n_splits=10)
        skf.get_n_splits(self.X_train,self.y_train)
        best_alpha=0
        max_accuracy=np.NINF
        min_rmse=np.inf
        if(function=='gda'):
            for alpha in hyper_list: 
                accuracy_array=[]
                iteration=0
                for train_index, test_index in skf.split(self.X_train, self.y_train):
                    iteration=iteration+1
                    X_train, X_test = self.X_train[train_index], self.X_train[test_index]
                    y_train, y_test = self.y_train[train_index], self.y_train[test_index]                 
                    w=self.gradient_ascent(X_train,y_train,X_train.shape[0],X_train.shape[1],alpha)
                    ypred=self.prediction_function(X_test,y_test,w)
                    tp,fp,fn,tn=self.performance_measures(y_test,ypred)
                    sensitivity,specificity,precision,accuracy,f_measure=self.get_final_performance(tp,fp,tn,fn)
                    accuracy_array.append(f_measure)
                avg_accuracy=np.average(accuracy_array)
                if(avg_accuracy>max_accuracy):
                    max_accuracy=avg_accuracy
                    best_alpha=alpha              
            print(best_alpha)
            return (best_alpha)
        elif(function=='gd'):
            for alpha in hyper_list: 
                rmse_array=[]
                iteration=0
                for train_index, test_index in skf.split(self.X_train, self.y_train):
                    iteration=iteration+1
                    X_train, X_test = self.X_train[train_index], self.X_train[test_index]
                    y_train, y_test = self.y_train[train_index], self.y_train[test_index]                 
                    w=self.gradient_descent(X_train,y_train,X_train.shape[0],X_train.shape[1],alpha)
                    rmse=self.find_rmse_gradient_descent(X_test,y_test,w)
                    rmse_array.append(rmse)
                avg_rm=np.average(rmse_array)
                if(avg_rm>min_rmse):
                    min_rmse=avg_rm
                    best_alpha=alpha              
            print(best_alpha)
            return (best_alpha)
        else:
            print('Function given is not correct. Retry')
            return
     
    def sigmoid(self,temp):
        return(1/(1+np.exp(-temp)))
        
    def decision_boundary():
        pass
        
    def plot_roc(self,ypred):
        #print(self.y_test,ypred)
        ytrue=np.array(self.y_test,dtype='uint8')
        ypred=np.array(ypred,dtype='uint8')
        y=[]
        for i in ypred:
            y.append(i[0])
        ypred=np.array(y,dtype='uint8')    
        fpr, tpr, thresholds=metrics.roc_curve(ytrue, ypred)        
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr,tpr)
        #plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        
    def __init__(self,hyperlist): 
        self.X, self.y=self.preprocess(0)
        self.hyperlist=hyperlist
        self.attributes=self.X.shape[1]
        self.X_train, self.X_test, self.y_train, self.y_test=self.train_test_splitter()
        self.train_rows=self.X_train.shape[0]
        self.w=self.gradient_ascent(self.X_train,self.y_train,self.train_rows,self.attributes,hyperlist[4])
        self.ypred=self.prediction_function(self.X_test,self.y_test,self.w)
        self.tp,self.fp,self.fn,self.tn=self.performance_measures(self.y_test,self.ypred)
        self.sen,self.spe,self.pre,self.acc,self.f_m=self.get_final_performance(self.tp,self.fp,self.tn,self.fn)

hyperlist=[]
for alpha in range(-8,-2):
    hyperlist.append(10**alpha)  
      
logistic_reg=Classifier(hyperlist)

print(logistic_reg.sen,logistic_reg.spe,logistic_reg.acc,logistic_reg.f_m)
#Calling via cross validation: Logistic Regression
#alpha=logistic_reg.cross_validator(hyperlist,'gda')
#w=logistic_reg.gradient_ascent(logistic_reg.X_train,logistic_reg.y_train,logistic_reg.train_rows,logistic_reg.attributes,alpha)
#ypred=logistic_reg.prediction_function(logistic_reg.X_test,logistic_reg.y_test,w)
#logistic_reg.plot_roc(ypred)
#tp,fp,fn,tn=logistic_reg.performance_measures(logistic_reg.y_test,ypred)
#sen,spe,pre,acc,f_m=logistic_reg.get_final_performance(tp,fp,tn,fn)
#print(logistic_reg.sen,logistic_reg.spe,logistic_reg.acc,logistic_reg.f_m)
print('Done')
#logistic_reg.cross_validator(hyperlist,'gda')

class Regression(Classifier):
    
    def train_test_splitter(self):
         print("overrided")
         X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.3)
         return X_train, X_test, y_train, y_test

    def find_rmse_gradient_descent(self,X_test,y_test,w):
        sum=0
        f_x=np.dot(X_test,w)
        for i in range(0,len(X_test)):
            sum=sum+((f_x[i]-y_test[i])**2)        
        rmse=math.sqrt(sum/len(X_test))
        return rmse
    
    def __init__(self,hyperlist,attri): 
        self.X, self.y=self.preprocess(attri)
        print(self.X)
        self.hyperlist=hyperlist
        self.attributes=self.X.shape[1]
        self.X_train, self.X_test, self.y_train, self.y_test=self.train_test_splitter()
        self.train_rows=self.X_train.shape[0]
        self.w=self.gradient_descent(self.X_train,self.y_train,self.train_rows,self.attributes,hyperlist[2])
        self.rmse=self.find_rmse_gradient_descent(self.X_test,self.y_test,self.w)
        
    
linear=Regression(hyperlist,3)
print(linear.rmse)