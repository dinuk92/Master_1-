# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 10:47:54 2017

@author: 3200183
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from arftools import *
import random as rn
from  sklearn import  linear_model
from sklearn import svm



def hinge(datax,datay,w,lb):
    return np.mean(np.maximum(0,-np.dot(datay,np.dot(datax,w)))+ lb * np.dot(w,w.T))

# gradient
def hinge_grad(datax,datay,w,lb):
     
    #somme=np.zeros((len(w)))
    
       # if (np.sign(datay[i]*np.dot(datax[i],w))[0]==-1):
    c=((datay*np.dot(datax,w)) > 0) * datay
       
    return -np.mean(np.dot(c,datax))+2*lb*w  
    
def v2m(x):
    return x.reshape ((1 ,x.size)) if len(x.shape)==1 else x
    
def v2m2(x):
    return x.reshape ((x.size,1)) if len(x.shape)==1 else x
    
class Perceptron(object):
    def __init__(self,loss=hinge,loss_g=hinge_grad,p=None,max_iter=100,eps=0.01, lb=0):
        self.max_iter, self.eps = max_iter,eps
        self.w = None
        self.lb = lb
        self.w_histo,self.loss_histo = None,None
        self.loss = loss
        self.p=p
        self.loss_g = loss_g

    def fit(self,datax,datay):
        #datay = datay.reshape(-1,1)
        N = len(datay)
        #datax = datax.reshape(N,-1)
        
        if (self.p== None ):
            print "test1"
            datax=np.hstack((np.ones((N,1)),datax.reshape(N,-1)))
        elif (self.p == "prog3D")  :
            print "test2"
            d3 = v2m2(np.array(datax[:,1]*datax[:,0]))
                        
            print d3.shape
            print datax.reshape(N,-1).shape
            print np.ones((N,1)).shape
            
            datax=np.hstack((datax.reshape(N,-1),d3   ))
            datax=np.hstack((np.ones((N,1)),datax.reshape(N,-1)))
            
        D = datax.shape[1]
        print D
        self.w = np.random.random((1,D))
        self.w = self.w.T
        #self.w = np.array(self.w)
        self.w_histo = []
        self.loss_histo = []
        
        for i in range(self.max_iter):
            f = self.loss(datax, datay, self.w , self.lb)
            
            
            self.w_histo.append(self.w)
            self.loss_histo.append(f)
            a=v2m2(self.loss_g(datax, datay, self.w,self.lb))
            
            #a=self.loss_g(datax, datay, self.w).reshape(3,1)   
            self.w = self.w - self.eps * a    
           
        return (self.w_histo, self.loss_histo)
            
    def predict(self,datax):
        N = len(datax)
        if (self.p== None ):
            print "test1"
            datax=np.hstack((np.ones((N,1)),datax.reshape(N,-1)))
        elif (self.p == "prog3D")  :
            print "test2"
            d3 = v2m2(np.array(datax[:,1]*datax[:,0]))
                        
            print d3.shape
            print datax.reshape(N,-1).shape
            print np.ones((N,1)).shape
            
            datax=np.hstack((datax.reshape(N,-1),d3   ))
            datax=np.hstack((np.ones((N,1)),datax.reshape(N,-1)))
        liste=[]
        for i in datax:            
            liste.append( np.dot(v2m(i),self.w ) )
        return np.array(liste)
        
    def score(self,datax,datay):
        return np.mean(self.predict(datax)==datay)    


(x1, y1) = gen_arti(centerx=1, centery=1, sigma=0.05, nbex=100, data_type=0, epsilon=0.02)
(x2, y2) = gen_arti(centerx=1, centery=1, sigma=0.05, nbex=100, data_type=0, epsilon=0.02)

'''
a = linear_model.Perceptron(penalty='elasticnet')
a.fit(x1,y1)
y=  a.predict(x2)
print a.score(x2,y)
plt.figure(1)
plot_frontiere(x2, a.predict,step=20)
plot_data(x2, y2)


plt.figure(2)

clf = svm.SVC(kernel='linear')
clf.fit(x1, y1)
plot_frontiere(x2, clf.predict,step=20)
plot_data(x2, y2)

p = Perceptron(loss=hinge,loss_g=hinge_grad, max_iter=100, eps=0.01,lb=100)
# Pour le cas stockastique
#p = Perceptron(loss=hinge,loss_g=hinge_grad_s, max_iter=100, eps=0.01)

(log_w, log_f) = p.fit(x1,y1)
plt.figure(3)
plot_frontiere(x2, p.predict,step=20)
plot_data(x2, y2)
'''

clf = svm.SVC( probability = True ,kernel='linear')
clf.fit(x1, y1)

#plot_frontiere(x2, clf.predict,step=20)
#plot_data(x2, y2)
plt.figure()
def  plot_frontiere_proba ( data , f , step =20):
    grid , x , y=make_grid( data=data , step=step )
    plt.contourf (x,y, f(grid).reshape(x.shape),255)


plot_frontiere_proba(x2,lambda x:clf.predict_proba(x)[:,0] , step=50)
plot_data(x2, y2)

res = clf.support_vectors_
print res
print res[:,0]
plt.scatter(res[:,0],res[:,1],marker='x', color='m')



def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def show_usps(data):
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")
    
    
datax , datay = load_usps("USPS_train.txt")  

show_usps(datax[0])

