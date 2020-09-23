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


model = NBCC()

model.fit(features_TR,target_TR)

pred = model.predict(features_TR)

print(pred)