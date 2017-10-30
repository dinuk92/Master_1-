# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 11:22:19 2017

@author: 3200183
"""

print(__doc__)

from sklearn import clone
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
from arftools import *
model = Perceptron()
model.n_iter = 1000
#model = DecisionTreeClassifier()
#model.max_depth=3

(X,y) =   gen_arti(centerx=1, centery=1, sigma=0.1, nbex=1000, data_type=1, epsilon=0.02)

d= np.ones(len(X))/len(X)

print len(d) 
print len(y)

res = []
eps = 0.1
step=0
while eps <0.5 and eps>0 and step<100:
    print(step)
    clf = clone(model)
    clf.fit(X, y , sample_weight = d  )
    ht =clf.predict(X)
    #print "test", np.where(ht==y,0 , 1)*d
    eps= np.sum(np.where(ht==y,0 , 1)*d)
    if eps>0.5 or eps<=0:
        break
    print "test",eps
    alpha = np.log((1-eps)/eps)/2
    #print alpha
    e =np.array([np.exp(-alpha*y[i]*ht[i]) for i in np.arange(len(y))]) 
    #print e
    d=np.array([d[i]*e[i] for i in np.arange(len(X))])
    #print d
    z= np.sum(d)
    #print z
    d= d/z
    #print d
    res.append((alpha, clf))
    step+=1



class prediction(object):
    def __init__(self,res):
        self.res = res
    
    def predict(self,x):
        p = 0
        for i in self.res : 
            p =+ i[0] * i[1].predict(x)*1.0
        return p
        
        

classif = prediction(res)
make_grid(X)
plot_frontiere(X,classif.predict)
plot_data(X,y)
