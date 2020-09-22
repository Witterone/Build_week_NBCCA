# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 11:51:18 2020

@author: Ronin
"""
import numpy as np
def NBCC():
    """ Naive Bayesian Categorical Classifier,
    Given a your array of target data and 2d array of features for training
    it should be able to predict a target array based on testing features.
    """
    def __init__(self):
        self.target = []
        self.features = []
        self.length = 0
        
        
        #need to get target in array and use np.unique
        #find probability of each unique value
        #
    def fit(self,features,target):
        self.target = np.array(target)
        self.features = np.array(features)
        self.length = len(self.target)
        self.targetUS = set(np.unique(self.target))
        
        for i in self.targetUS:
            x = prob_find(i,self.target)
            self.targetprob.append(x)
        self.target_data = {}
        for z in self.targetUS:
            for q in self.targetprob:
                self.target_data[z] = q
        
        self.featcube = []
        for i in self.features.T:
            f = feature(i)
            self.featcube.append(f)
       
        
        """
        find the highest probability value of a target value given any unique 
        feature value, then for predict add matching target value probabilities
        to determine most likely value for that row.
        """
    def feature(self,data):
        self.data = data
        self.probGtarget = []
        self.u_features = set(np.unique(self.data))
        for i in self.u_features:
            u = prob_find(i,self.data)
            self.featureprob.append(u)
        self.featdata = {}
        for a in self.u_features:
            for b in self.featureprob:
                self.featdata[a] = b
        for q in self.u_features:
            for r in range(self.targetUS):
                    self.pFgT[q][self.targetUS[r]]=self.probGtarget[r]
        
        self.PTGF = []
        for t in self.u_features:
            max_v = 0
            for v in self.targetUS:
                probop = bayes(self.target_data[v],
                               self.featdata[t],self.pFgT[t][v])
                if probop > max_v:
                    max_v = probop
                    t_v = v
                    
            self.PTGF.append(t_v)
        
        for p in self.u_features:
            for y in self.PTGF:
                self.TSGFWP[p] = y
                    
                
                
    def target_box(self,value):
        self.v_prob = prob_index(value,self.target_data)
        for i in range(len(self.target)):
            target_features = []
            if self.target[i] == value:
               target_features.append(self.features[i])
               
        target_features = target_features.T
        length = len(target_features)
        for i in range(length):
            for x in self.featcube[i].u_features:
                u = prob_find(x,target_features[i])
                self.featcube[i].probGtarget.append(u)
        
                
    def bayes(self,Ptarget_v,Pfeature_v,pfgt):
        prob = (pfgt*Ptarget_v)/Pfeature_v
        return prob
         
   
        
    def predict(self,test_data):
        final_prediction = []
        for row in test_data:
            size = len(self.featcube)
            pred_votes =[]
            for ln in range(size):
                f_v = row[ln]
                vote = self.featcube[ln].TSGFWP[f_v]
                pred_votes.append(vote)
            
            prediction = most_frequent(pred_votes)
            final_prediction.append(prediction)
            
        return final_prediction
        
            
    def most_frequent(List): 
        return max(set(List), key = List.count)           
                
    def prob_index(self,value, data):
        return data[value]
        
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