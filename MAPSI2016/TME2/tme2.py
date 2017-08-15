# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 10:47:56 2016

@author:  3200183
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
        if(x <= p):
            return 1
        else:
            return 0
    return -1

print "Bernouilli"    
print bernoulli(0.7)


""" I.2. Loi binomiale """

def binomiale (n,p) :
    if(p >= 0 and p <= 1):
        cpt = np.array([bernoulli(p) for i in range(n)])
        res = sum(cpt)
        return res
    return -1    

print "Binomiale"
print binomiale(10,0.6)

""" I.3. Histogramme de la loi binomiale """

def galton(hauteur,nbInstance):
    tabValeur = np.zeros(nbInstance, int)
    tabOccurence = np.zeros(hauteur, int)
    intervalle = 0;
    i = 0
    for i in range(nbInstance):                    
        tabValeur[i] = binomiale(hauteur,0.5)
        tabOccurence[tabValeur[i]] += 1
    for i in range(hauteur): 
       if(tabOccurence[i] > 0):
            intervalle += 1
    print len(tabValeur)
    print len(tabOccurence)
    print intervalle
    return (tabValeur, intervalle)

"""
def galton2 (hauteur,nbInstance):
    tabValeur = np.array(binomiale(hauteur,0.5) in range(nbInstance))
    tabOccurance = np.array(cpt) for cpt in    
"""
       
tableau , n  =galton(20,1000)

plt.hist(tableau, n)
plt.show()


""" II. Visualisation d'indépendances """

""" II.1. Loi normale centrée """

def normale(k,sigma):
    if k % 2 == 0:
        raise ValueError ( 'le nombre k doit etre impair' )
        
    x = np.linspace(-2*sigma, 2*sigma, k)
    tab = np.array([(1/m.sqrt(2*m.pi)*sigma)*m.exp((-1/2)* pow((x[i]/sigma), 2)) for i in range(len(x))])
    return (x ,tab)
    
x,tab =normale(1001,100)  
plt.plot(x,tab)
plt.show()

x,tab =normale(1001,50)  
plt.plot(x,tab)
plt.show()

x,tab =normale(1001,30)  
plt.plot(x,tab)
plt.show()

    
""" II.2. Distribution de probabilité affine """    
    
def proba_affine(k, slope) :
    if k % 2 == 0:
        raise ValueError ( 'le nombre k doit etre impair' )
    if abs ( slope  ) > 2.0 / ( k * k ):
        raise ValueError ( 'la pente est trop raide : pente max = ' + 
        str ( 2.0 / ( k * k ) ) )
    tab = np.array([(1.0/k) + (i - (k-1)/2)*slope for i in range(k)] )
    return tab
    
tab = proba_affine(7,0.03 )      
plt.plot(tab)
plt.show()

tab = proba_affine(11,0.01 )      
plt.plot(tab)
plt.show()

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

def dessine ( P_jointe ):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace ( -3, 3, P_jointe.shape[0] )
    y = np.linspace ( -3, 3, P_jointe.shape[1] )
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, P_jointe, rstride=1, cstride=1 )
    ax.set_xlabel('A')
    ax.set_ylabel('B')
    ax.set_zlabel('P(A) * P(B)')
    plt.show ()

dessine(pyx(normale(11,12)[1],proba_affine(11,0.0032)))

""" III.2. Création d'une probabilité conditionnelle  """  
   
def conditional_proba(p):
    tab = [p[0][len(p[0])-1]]      
    p2 = marginalize_joint_proba(p,tab)
    res = divide_joint_probas(p,p2)
    return res
  
print conditional_proba(([2, 3, 5],np.array([0.03, 0.0015, 0.27, 0.1485, 0.02, 0.0035, 0.18, 0.3465])))

""" III. 3. Test d'indépendance conditionnelle"""

def is_last_var_conditionally_indep(p,xj):    
    proba_cond = conditional_proba(p)
    p = marginalize1Var(p,xj)
    p = conditional_proba(p)    
    return equal_conditional_probas(proba_cond,p)

print is_last_var_conditionally_indep(([2, 3, 5],np.array([0.03, 0.0015, 0.27, 0.1485, 0.02, 0.0035, 0.18, 0.3465])),2)

print is_last_var_conditionally_indep(([2, 3, 5],np.array([0.03, 0.0015, 0.27, 0.1485, 0.02, 0.0035, 0.18, 0.3465])),3)


""" III.4. Compactage de probabilités conditionnelles """  

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
X = create_bayesian_network(P)    
#print X


