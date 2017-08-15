# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 10:47:30 2016

@author: 3200183
"""


import numpy as np
from math import *
import matplotlib.pyplot as plt
import pickle as pkl

# truc pour un affichage plus convivial des matrices numpy
np.set_printoptions(precision=2, linewidth=320)
plt.close('all')

data = pkl.load(file("lettres.pkl","rb"))
X = np.array(data.get('letters'))
Y = np.array(data.get('labels'))
#print Y
nCl = 26
K = 10
N = 5

""" 1. Discrétisation """

def discretise(X,d) :
    discr = []
    intervalle = 360.0/d
    for x in X:
        discr.append(np.floor(x/intervalle))
    return np.array(discr)
    
def initGD(X,N):

    Xd = discretise(X,K)
    S = []    
    for x in Xd:
        S.append(np.floor(np.linspace(0,N-.00000001,len(x))))
    
    return (Xd,S)
    
Xd,q = initGD(X,N)

    
def learnHMM(allx, allq, N, K, initTo0=True):
    if initTo0:
        A = np.zeros((N,N))
        B = np.zeros((N,K))
        Pi = np.zeros(N)
    else:
        eps = 1e-8
        A = np.ones((N,N))*eps
        B = np.ones((N,K))*eps
        Pi = np.ones(N)*eps
      
    for x,q in zip(allx,allq):
        Pi[q[0]] += 1
        B[q[0],x[0]] += 1
        for i in np.arange(0, len(q) - 1):
            A[q[i],q[i+1]] += 1
            B[q[i+1],x[i+1]] += 1
            
    B = B/np.maximum(B.sum(1).reshape(N,1),1)    
    A = A/np.maximum(A.sum(1).reshape(N,1),1) # normalisation
    Pi = Pi/Pi.sum()  
        
    return (Pi,A,B)

newXd = []
newQ = []
ind = np.where(Y=='a')
for i in range(len(ind[0])):
    newXd.append(Xd[i])
    newQ.append(q[i])

Pi,A,B = learnHMM(newXd,newQ,N,K)
#print Pi
#print A
#print B



def viterbi(x,Pi,A,B):
    N = np.shape(B)[0]
    T = np.shape(x)[0]
    D = np.zeros((N,T))
    phi = np.zeros((N,T))
    S = np.zeros(T)
    
    #init
    for i in range(N):
        D[i,0] = np.log(Pi[i]) + np.log(B[i,x[0]])  
        phi[i,0] = -1
    
    #recursion
    for t in range(1,T):
        for j in range(N):
            D[j,t] = np.max(D[:,t-1]+np.log(A[:,j])) + np.log(B[j,x[t]])
            phi[j,t] = np.argmax(D[:,t-1]+ np.log(A[:,j]))

    #terminaison
    P = np.max(D[:,T-1])

    #chemin
    S[T-1] = np.argmax(D[:,T-1])
    for t in range(T-2,0,-1):
        S[t] = np.max(phi[S[t+1],t+1])
    return S,P

S,P = viterbi(Xd[0], Pi, A, B)

#print P
#print S

#print Xd
#print Xd[0]



def baum_welch(X, N):
    Xd , q  = initGD(X,N)   
    newXd = []
    newQ = []
    logproba = []
    while True :           
        sommeLettre = 0
        for l in range(nCl):            
            ind = np.where(Y== chr(97+l))
            for i in range(len(ind[0])):
                newXd.append(Xd[i])
                newQ.append(q[i])
            Pi,A,B = learnHMM(newXd,newQ,N,K)
            sommeviterbi = 0
            for i in range (len(newXd)):
               seq,proba = viterbi(newXd[i],Pi,A,B)
               sommeviterbi = sommeviterbi + proba
            sommeLettre = sommeLettre +  sommeviterbi
        
        if( len(logproba)> 2 ):
            if (logproba[-1]-logproba[-2])/logproba[-1] < 0.0001 :
                return logproba
        print sommeLettre
        logproba.append(sommeLettre)

LKtab = baum_welch(X,N)
print LKtab
plt.figure()
plt.plot(range(len(LKtab)),LKtab)