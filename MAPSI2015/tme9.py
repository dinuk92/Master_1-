# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 10:53:30 2015

@author: 3200386
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import numpy.random as npr

(count, mu, A) = pkl.load(file("countWar.pkl", "rb"))
secret = (open("secret.txt", "r")).read()[0:-1] # -1 pour supprimer le saut de ligne
secret2 = (open("secret2.txt", "r")).read()[0:-1] # -1 pour supprimer le saut de ligne


def tirage(m):
    return np.random.uniform(-m,m,2)
    
  
def monteCarlo(N):
    x = np.zeros(N)
    y = np.zeros(N)
    cpt = 0    
    for i in range(N):
        x[i] ,y[i] = tirage(1)
        if np.sqrt(x[i]**2 + y[i]**2) <= 1:
            cpt += 1
            
    return (4*cpt)/N, x, y    

"""
plt.figure()

# trace le carré
plt.plot([-1, -1, 1, 1], [-1, 1, 1, -1], '-')

# trace le cercle
x = np.linspace(-1, 1, 100)
y = np.sqrt(1- x*x)
plt.plot(x, y, 'b')
plt.plot(x, -y, 'b')

# estimation par Monte Carlo
pi, x, y = monteCarlo(int(1e4))

# trace les points dans le cercle et hors du cercle
dist = x*x + y*y
plt.plot(x[dist <=1], y[dist <=1], "go")
plt.plot(x[dist>1], y[dist>1], "ro")
plt.show()
"""


tau = {'a' : 'b', 'b' : 'c', 'c' : 'a', 'd' : 'd' }    
    
def swapF(tau):
    
    
    i = 0
    j = 0
    while(i == j):    
    
        i = np.random.randint(0, len(tau))
        j = np.random.randint(0, len(tau)) 
    
    v1 = tau.values()[i]
    v2 = tau.values()[j]
    
    tau[tau.keys()[i]] = v2
    tau[tau.keys()[j]] = v1
    
    return tau

#print swapF(tau)


def decrypt(mess,tau):
    chaine = ""
    for i in range(len(mess)):
        chaine = chaine + str(tau[mess[i]])
    return chaine


#print decrypt ( "aabcd", tau )
#print decrypt ( "dcba", tau )


def logLikelihood(mess,mu,A,chars):
    L = np.log(mu[chars.index(mess[0])])
    for i in range(len(mess)-1):
        ci = chars.index(mess[i])
        ci1 = chars.index(mess[i+1])
        L +=  np.log(A[ci,ci1])
    return L
    
print logLikelihood( "abcd", mu, A, count.keys () )
print logLikelihood( "dcba", mu, A, count.keys () )


def MetropolisHastings(mess,mu,A,tau,N,chars):
    Lmax = float("-inf")
    bestMess = ""
    for i in range(N):
        tauInter = swapF(tau)
        messdec = decrypt(mess,tauInter)
        
        L = logLikelihood(mess,mu,A,chars)
        Ls = logLikelihood(messdec,mu,A,chars)    
        rapport =  (L - Ls)
        print i        
        
        if rapport < 0 :
            alpha = rapport
        else :
            alpha = 0
            
        if  np.random.uniform(0,1) <= alpha * -1 :
            mess = messdec
            tau = tauInter
            if Lmax < Ls:
                Lmax = Ls
                bestMess = messdec
    return (bestMess,Lmax)
    
def identityTau ():
    tau = {}
    for k in count.keys ():
        tau[k] = k
    return tau
    
#print MetropolisHastings( secret2, mu, A, identityTau (), 10000,count.keys() )

def updateOccurrences(text, count):
   for c in text:
      if c == u'\n':
         continue
      try:
         count[c] += 1
      except KeyError as e:
         count[c] = 1

def mostFrequent(count):
   bestK = []
   bestN = -1
   for k in count.keys():
      if (count[k]>bestN):
         bestK = [k]
         bestN = count[k]
      elif (count[k]==bestN):
         bestK.append(k)
   return bestK

def replaceF(f, kM, k):
   try:
      for c in f.keys():
         if f[c] == k:
            f[c] = f[kM]
            f[kM] = k
            return
   except KeyError as e:
      f[kM] = k

def mostFrequentF(message, count1, f={}):
   count = dict(count1)
   countM = {}
   updateOccurrences(message, countM)
   while len(countM) > 0:
      bestKM = mostFrequent(countM)
      bestK = mostFrequent(count)
      if len(bestKM)==1:
         kM = bestKM[0]
      else:
         kM = bestKM[npr.random_integers(0, len(bestKM)-1)]
      if len(bestK)==1:
         k = bestK[0]
      else:
         k = bestK[npr.random_integers(0, len(bestK)-1)]
      replaceF(f, kM, k)
      countM.pop(kM)
      count.pop(k)
   return f



tau_init = mostFrequentF(secret2, count, identityTau () )

print MetropolisHastings(secret2, mu, A, tau_init, 10000,count.keys() )

