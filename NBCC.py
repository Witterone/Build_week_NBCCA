# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 11:51:18 2020

@author: Ronin
"""
import numpy as np
class NBCC():
    """ Naive Bayesian Categorical Classifier,
    Given a your 1d array of target data and 2d array of features for training
    it should be able to predict a target array based on testing features.
    Use "fit" to train the model and "predict" to let the model make 
    predictions based on the training data.
    """
    def __init__(self):
        self.target = []
        self.features = []
        self.length = 0
        self.target_data = {}
        self.TSGFWP = {}
        self.targetUS = None
        self.probGtarget = {}
        
    def prob_find(self,targetcon, arr):
        
        if self.length == 0:
            return "Invalid Data size"
        if targetcon not in arr or targetcon is None:
            return 0
        counter = 0
        for i in arr:
             if i == targetcon:
                 counter +=1
                 
        return counter/len(arr)
        
        
    def fit(self,features,target):
        """fitting the model to your data given 1d -2d features and 
        1d target data. 
        
        """
        self.target = np.array(target)
        self.features = np.array(features)
        self.length = len(self.target)
        self.targetUS = set(np.unique(self.target))
        targetprob = []
        for i in self.targetUS:
            x = self.prob_find(i,self.target)
            targetprob.append(x)
        
        for z in self.targetUS:
            for q in targetprob:
                self.target_data[z] = q
        
        
        for i in self.features.T:
             self.feature(i)
            
            
      
        
    def feature(self,data):
        featureprob = []
        
        u_features = set(np.unique(data))
        for i in u_features:
            u = self.prob_find(i,data)
            featureprob.append(u)
        featdata = {}
        for a in u_features:
            for b in featureprob:
                featdata[a] = b
        for x in self.targetUS:
            self.Given_Target(x,u_features,data)
        
        PTGF = []
        for t in u_features:
            max_v = 0
            for v in self.targetUS:
                probop = self.bayes(self.target_data[v],
                                featdata[t],self.probGtarget[t][v])
                if probop > max_v:
                    max_v = probop
                    t_v = v
                    
            PTGF.append(t_v)
        
        for p in u_features:
            for y in PTGF:
                self.TSGFWP[p] = y
                    
                
                
    def Given_Target(self,value,u_features,a_features):
        target_features = []
        for i in range(len(self.target)):
            
            if self.target[i] == value:
               target_features.append(a_features[i])
               
     
        length = len(target_features)
        for i in range(length):
            for x in u_features:
                u = self.prob_find(x,target_features[i])
                self.probGtarget[x][value] = u
        
                
    def bayes(self,Ptarget_v,Pfeature_v,pfgt):
        prob = (pfgt*Ptarget_v)/Pfeature_v
        return prob
         
   
        
    def predict(self,test_data):
        final_prediction = []
        for row in test_data:
            size = len(row)
            pred_votes =[]
            for ln in range(size):
                f_v = row[ln]
                vote = self.TSGFWP[f_v]
                pred_votes.append(vote)
            
            prediction = self.most_frequent(pred_votes)
            final_prediction.append(prediction)
            
        return final_prediction
        
            
    def most_frequent(List): 
        return max(set(List), key = List.count)           
                
        
    