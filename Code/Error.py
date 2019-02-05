# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 10:26:42 2018

@author: naz2hi
"""

import numpy as np
import pandas as pd

class err:    
    def err(a,b):
        org=pd.DataFrame()
        org['test']=a
        org['Pred']=b
        for i in range(0,65):
            temp=org[org['test']==i]
            if len(temp)!=0:
                print(i,np.mean(temp['test']==temp['Pred']))
                
    def class_err(a,b):
        org=pd.DataFrame()
        org['test']=a
        org['Pred']=b
        comp=list(a.unique())
        for each in comp:
            temp=org[org['test']==each]
            if len(temp)!=0:
                print(each,np.mean(temp['test']==temp['Pred']))