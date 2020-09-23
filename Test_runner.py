# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 12:07:41 2020

@author: Ronin
"""
from NBCC import NBCC


target_TR = ["blue","red","blue","white","red","white"]

features_TR =[["car","truck","car","van","truck","van"],
              ["Ford","Chevy","Ford","Chevy","Chevy","Chevy"],
              ["New","Used","Used","New","New","Used"]]

features_TB =[["car","Ford","New"],
              ["truck","Chevy","Used"],
              ["car","Ford","Used"],
              ["van","Chevy","New"],
              ["truck","Chevy","New"],
              ["van","Chevy""Used"]]
model = NBCC()

model.fit(features_TB,target_TR)

pred = model.predict(features_TB)

print(pred)