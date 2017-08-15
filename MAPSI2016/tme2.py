# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 09:39:49 2016

@author: 3200183
"""

import numpy as np 
import matplotlib.pyplot as plt
import math as m 
from mpl_toolkits.mplot3d import Axes3D

def bernoulli(p):
    if(p >= 0 and p <= 1 ):
       r =np.random.random()
       if (r <= p):
           return 1;
       else:
           return 0;
    return -1
    
  
def binomiale(n, p):
    cpt = 0
    if (bernoulli(p) == -1) :
        return -1;
    else:
        for i in range(n):
            b = bernoulli(p)
            #print b 
            cpt = cpt + b;
        return cpt
        

print binomiale(3,0.5)



def galton(hauteur):
    tabV = np.zeros(1000)
    tabO = np.zeros(1000)
    intervalle = 0    
    for i in range(1000):
        tabV[i]= binomiale(hauteur,0.5)
        tabO[tabV[i]] += 1  
    for i in range(1000):
        if(tabO[i]>0):
            intervalle += 1
    plt.hist(tabV,intervalle)
    plt.show()

galton(10)

def normale ( k, sigma ):
    if k % 2 == 0:
        raise ValueError ( 'le nombre k doit etre impair' )
    x= np.linspace(-2* sigma, 2*sigma)
    tab = np.zeros(len(x))    
    for i in range(len(x)):
            tab[i] = (1/m.sqrt(2*m.pi)*sigma)*m.exp((-1/2)* pow((x[i]/sigma),2))
    return tab 

x = (1/m.sqrt(2*m.pi)*12)*m.exp((-1/2)* pow((0/12),2))
print x

plt.plot(normale(51,12))
plt.plot()

        
    
    
    
    