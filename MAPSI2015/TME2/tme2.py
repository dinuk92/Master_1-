# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 22:25:28 2015

@author: 3200386
"""

import numpy as np
import matplotlib.pyplot as plt
import math as m
from mpl_toolkits.mplot3d import Axes3D
from probabilite import *
import copy
plt.close('all') 


""" I. La planche de Galton """

""" I.1. Loi de Bernoulli """

def bernoulli(p):
    if(p >= 0 and p <= 1):
        x = np.random.random()
        #print "random %f" %x
        if(x <= p):
            return 1
        else:
            return 0
    return -1        
    
print "bernoulli %d" %bernoulli(0.7)


""" I.2. Loi binomiale """
def binomiale (n,p) :
    if(p >= 0 and p <= 1):
        cpt = 0
        for i in range(n):
            cpt = cpt + bernoulli(p)
        return cpt
    return -1    

print "binomiale %d" %binomiale(10,0.6)
    

""" I.3. Histogramme de la loi binomiale """

def galton(nbOccurence,nbInstance):
    tabValeur = np.zeros(nbInstance, int)
    tabOccurence = np.zeros(nbOccurence, int)
    intervalle = 0;
    i = 0
    for i in range(nbInstance):                    
        tabValeur[i] = binomiale(nbOccurence,0.5)
        tabOccurence[tabValeur[i]] += 1
    for i in range(nbOccurence): 
       if(tabOccurence[i] > 0):
            intervalle += 1
            
    #print tabValeur        
    #print tabOccurence
    #print intervalle
    
    plt.hist(tabValeur, intervalle)
    plt.show()
        
print galton(20, 1000)

""" II. Visualisation d'indépendances """

""" II.1. Loi normale centrée """

def normale(k,sigma):
    if k % 2 == 0:
        raise ValueError ( 'le nombre k doit etre impair' )
        
    x = np.linspace(-2*sigma, 2*sigma, k)
    tab = np.zeros(len(x))
    
    for i in range (len(x)):
        tab[i]= (1/m.sqrt(2*m.pi)*sigma)*m.exp((-1/2)* pow((x[i]/sigma), 2))
    
    plt.plot(x,tab)
    plt.show()    
    return tab
    
print normale(51,12)  

    
""" II.2. Distribution de probabilité affine """    
    
def proba_affine(k, slope) :
    if k % 2 == 0:
        raise ValueError ( 'le nombre k doit etre impair' )
    if abs ( slope  ) > 2.0 / ( k * k ):
        raise ValueError ( 'la pente est trop raide : pente max = ' + 
        str ( 2.0 / ( k * k ) ) )
    tab=np.zeros(k)
    for i in range (k) :
        tab[i]= (1.0/k) + (i - (k-1)/2)*slope
    plt.plot(tab)
    plt.show()
    return tab
    
print proba_affine(7,0.03 )      

""" II.3. Distribution jointe """  

def pyx (pa, pb):
      
    tab = np.zeros((len(pa),len(pb)),float)   
    
    for i in range(len(pa)):
        for j in range(len(pb)):
            tab[i,j] = pa[i] * pb[j]
            
    return tab

PA = np.array ( [0.2, 0.7, 0.1] )
PB = np.array ( [0.4, 0.4, 0.2] )
print pyx(PA,PB)
        
""" II.4. Affichage de la distribution jointe """  


def dessine ( P_jointe ):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace ( -3, 3, P_jointe.shape[0] )
    y = np.linspace ( -3, 3, P_jointe.shape[1] )
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(Y, X, P_jointe, rstride=1, cstride=1 )
    ax.set_xlabel('A')
    ax.set_ylabel('B')
    ax.set_zlabel('P(A) * P(B)')
    plt.show ()
    
print dessine(pyx(normale(11,12),proba_affine(11,0.0032)))    
  
""" III. Indépendances conditionnelles et consommation mémoire """  
  
""" III.1. Téléchargement d'un module de manipulation de probabilités """  

#Lu et approuve

""" III.2. Création d'une probabilité conditionnelle  """  
  

  
def conditional_proba(p):
    tab = [p[0][len(p[0])-1]]  
    
    p2 = marginalize_joint_proba(p,tab)
    res = divide_joint_probas(p,p2)
    return res
  
print conditional_proba(([2, 3, 5],np.array([0.03, 0.0015, 0.27, 0.1485, 0.02, 0.0035, 0.18, 0.3465])))
  

""" III.3. Test d'indépendance conditionnelle  """  

print "is_last_var_conditionally_indep"

def is_last_var_conditionally_indep(p,Xj):
    
    proba_cond = conditional_proba(p)
    p = marginalize1Var(p,Xj)
    p = conditional_proba(p)
    
    return equal_conditional_probas(proba_cond,p)
    
print is_last_var_conditionally_indep(([2, 3, 5],np.array([0.03, 0.0015, 0.27, 0.1485, 0.02, 0.0035, 0.18, 0.3465])),3)


""" III.4. Compactage de probabilités conditionnelles """  

print "compact_proba"

def compact_proba(p):
    
    K = copy.copy(p)

    for i in K[0]:
        if(is_last_var_conditionally_indep(K,i)):
            p = marginalize1Var(p,i)
            p = conditional_proba(p)  
  
    return p

print compact_proba(([2, 3, 5],np.array([0.03, 0.0015, 0.27, 0.1485, 0.02, 0.0035, 0.18, 0.3465])))


""" Partie Optionnelle """

""" III.5. Création d'un réseau bayésien """

     


def create_bayesian_network(p):
   
    liste = []  
   
    K = copy.deepcopy(p)
    
    for i,j in zip(range(len(K[0])-1,-1,-1),K[0]):
        print i
        Q = compact_proba(p)
        #print Q
        liste.append(Q)
        Q = marginalize1Var(Q,j)
        Q = conditional_proba(Q)

    return liste


P = read_file ( "2015_tme2_asia.txt" )
X = create_bayesian_network ( P)    
#print X
