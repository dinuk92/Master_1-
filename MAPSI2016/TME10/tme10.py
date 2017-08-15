# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 10:49:32 2016

@author: 3200183
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr


a = 6.
b = -1.
c = 1
N = 100
sig = .4 # écart type

def genererTab() :
    tabx = np.zeros(N)
    taby = np.zeros(N)
    for i in range (N) :
        tabx[i] = npr.rand()
        taby[i] = a*tabx[i] + b + (sig*npr.randn())
    return tabx,taby

tabx ,taby = genererTab()

plt.plot(tabx,taby,'ro')


def estimationProbabiliste():
    varx=np.var(tabx)   
    alpha= np.cov(tabx,taby)[0,1]/varx
    beta = np.mean(taby) - alpha * np.mean(tabx)
    return alpha,beta

alpha , beta = estimationProbabiliste()

plt.plot([0,1],[beta,alpha+beta])


def moindreCarre():
        
    X = np.hstack((tabx.reshape(N,1),np.ones((N,1))))
    Xt = np.transpose(X)
    Y = taby.reshape(N,1)
    A = np.dot(Xt,X)
    B = np.dot(Xt,Y)
    res = np.linalg.solve(A,B)
    
    return res

A,B = moindreCarre()

"""
plt.plot([0,1],[B,A+B])
plt.show()
"""
def descenteGradient():
    
    X = np.hstack((tabx.reshape(N,1),np.ones((N,1))))
    Xt = np.transpose(X)
    wstar = np.linalg.solve(X.T.dot(X), X.T.dot(taby)) # pour se rappeler du w optimal

    eps = 5e-4
    nIterations = 1000
    w = np.zeros(X.shape[1]) # init à 0
    allw = [w]
    for i in xrange(nIterations):
        # A COMPLETER => calcul du gradient vu en TD
        derivC = 2 * np.dot(Xt,np.dot(X,w)-taby)
        w = w - eps * derivC
        allw.append(w)
        #print w

    allw = np.array(allw)
    return (w,allw)

w , allw = descenteGradient()

def tracerGradient():
    
    # tracer de l'espace des couts
    ngrid = 20
    w1range = np.linspace(-0.5, 8, ngrid)
    w2range = np.linspace(-1.5, 1.5, ngrid)
    w1,w2 = np.meshgrid(w1range,w2range)
    X = np.hstack((tabx.reshape(N,1),np.ones((N,1))))
    wstar = np.linalg.solve(X.T.dot(X), X.T.dot(taby)) # pour se rappeler du w optimal
    

    cost = np.array([[np.log(((X.dot(np.array([w1i,w2j]))-taby)**2).sum()) for w1i in w1range] for w2j in w2range])

    plt.figure()
    plt.contour(w1, w2, cost)
    plt.scatter(wstar[0], wstar[1],c='r')
    plt.plot(allw[:,0],allw[:,1],'b+-' ,lw=2 )
 
tracerGradient()

def genererQuad():
    tabx = np.zeros(N)
    taby = np.zeros(N)
    for i in range (N) :
        tabx[i] = npr.rand()
        taby[i] = a*tabx[i]**2 + b*tabx[i]+ c + (sig*npr.randn())
    return tabx,taby
    
    
tabxquad ,tabyquad = genererQuad()    
#print tabxquad
#print tabyquad

plt.figure()
plt.plot(tabxquad,tabyquad,'ro')

 
def moindreCarreQuad():
    Xe = np.hstack(((tabxquad**2).reshape(N,1),tabxquad.reshape(N,1),np.ones((N,1))))
    #print Xe
    Xet = np.transpose(Xe)
    Y = tabyquad.reshape(N,1)
    A = np.dot(Xet,Xe)
    B = np.dot(Xet,Y)
    res = np.linalg.solve(A,B)
    return res

alpha , beta , delta = moindreCarreQuad()
yQuad = alpha*tabxquad**2 + beta * tabxquad + c
"""
print alpha
print beta
print delta

print alpha*tabxquad[0]**2 + beta * tabxquad[0] + c
print yQuad[0]
"""

tabxquad,yQuad = zip(*sorted(zip(tabxquad, yQuad)))
plt.plot(tabxquad,yQuad)
plt.show()