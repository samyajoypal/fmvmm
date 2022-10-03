#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 02:13:06 2022

@author: samyajoypal
"""
from mixtures.DMM_Class import DMM
from mixtures.DMM_Soft_Class import DMM_Soft

import numpy as np
import pandas as pd


np.random.seed(0)
s1 = pd.DataFrame(np.random.dirichlet((12, 12, 4), 500))
s2 = pd.DataFrame(np.random.dirichlet((12, 25, 55), 100))
s3 = pd.DataFrame(np.random.dirichlet((20, 20, 20), 300))
s4 = pd.DataFrame(np.random.dirichlet((0.25, 0.55, 4), 400))
initial_clusters=sum([[0]*500,[1]*100,[2]*300,[3]*400],[])



s=pd.concat([s1,s2,s3,s4], ignore_index= True)
s_copy=s
s_copy['Initial_Clusters'] = initial_clusters


s_shuffled= s_copy.sample(frac=1)
s_shuffled1=[s_shuffled[0],s_shuffled[1],s_shuffled[2]]
s_shuffled1=pd.DataFrame(s_shuffled1)
s_shuffled1=s_shuffled1.transpose() 

cluster32=DMM_Soft(4,s_shuffled1,method="fixediteration")
cluster32=DMM_Soft(4,s_shuffled1)
cluster32.fit()
cluster32.bic()


cluster33=DMM(4,s_shuffled1)
cluster33.fit()
cluster33.bic()
cluster33.predict()