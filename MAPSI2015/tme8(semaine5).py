# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 10:48:12 2015

@author: 3200386
"""

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

""" 1. Résultats de référence avec la méthode génératrice """

# fonction de suppression des 0 (certaines variances sont nulles car les pixels valent tous la même chose)
def woZeros(x):
    y = np.where(x==0., 1., x)
    return y

# Apprentissage d'un modèle naïf où chaque pixel est modélisé par une gaussienne (+hyp. d'indépendance des pixels)
# cette fonction donne 10 modèles (correspondant aux 10 classes de chiffres)
# USAGE: theta = learnGauss ( X,Y )
# theta[0] : modèle du premier chiffre,  theta[0][0] : vecteur des moyennes des pixels, theta[0][1] : vecteur des variances des pixels
def learnGauss (X,Y):
    theta = [(X[Y==y].mean(0),woZeros(X[Y==y].var(0))) for y in np.unique(Y)]
    return (np.array(theta))

# Application de TOUS les modèles sur TOUTES les images: résultat = matrice (nbClasses x nbImages)
def logpobs(X, theta):
    logp = [[-0.5*np.log(mod[1,:] * (2 * np.pi )).sum() + -0.5 * ( ( x - mod[0,:] )**2 / mod[1,:] ).sum () for x in X] for mod in theta ]
    return np.array(logp)

######################################################################
#########################     script      ############################


# Données au format pickle: le fichier contient X, XT, Y et YT
# X et Y sont les données d'apprentissage; les matrices en T sont les données de test
data = pkl.load(file('usps_small.pkl','rb'))

X = data['X']
Y = data['Y']
XT = data['XT']
YT = data['YT']

theta = learnGauss ( X,Y ) # apprentissage

logp  = logpobs(X, theta)  # application des modèles sur les données d'apprentissage
logpT = logpobs(XT, theta) # application des modèles sur les données de test

ypred  = logp.argmax(0)    # indice de la plus grande proba (colonne par colonne) = prédiction
ypredT = logpT.argmax(0)

print "Taux bonne classification en apprentissage : ",np.where(ypred != Y, 0.,1.).mean()
print "Taux bonne classification en test : ",np.where(ypredT != YT, 0.,1.).mean()

""" 2. Approche discriminante """

def fxi(x,w,b):
    res = 0
    
    for i in range(len(w)):
        res += x[i]*w[i]
    
    return 1/ (1+ np.exp(-(res + b)))

def Llog(x,w,b,y):
    
    
    res = 0
    for i in range(len(y)):
        res += y[i]*np.log(fxi(x,w,b)) + (1-y[i])*(np.log(1-fxi(x,w,b)))
    
    return res
    
def deriv_wj(x,w,b,y,j):
    
    res = 0
    
    for i in range(len(y)):
        res += x[i,j](y[i] - fxi(x,w,b))
        
    return res
    
def deriv_b(x,w,b,y):
    
    res = 0
    
    for i in range(len(y)):
        res += y[i] - fxi(x,w,b)
        
    return res

def apprentissage(x,y,classe):
    
    Yci = np.where(y==classe,1.,0.)
    epsilon = 0.00001
    
    w = np.random.uniform(-epsilon,epsilon,len(x[1]))
    b = np.random.uniform(-epsilon,epsilon,1)
    lkh = Llog(x,w,b,Yci)    
    
    
    for i in range(100):
        wp = w
        for j in range(len(w)):
            w[j] = w[j] + epsilon * deriv_wj(x,w,b,Yci,j)
        b = b + epsilon * deriv_b(x,wp,b,Yci)
    
    plt.figure()
    plt.plot(range(10),lkh,label='apprentissage')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    plt.show()

    return lkh,w,b
    
print apprentissage(X,Y,0)

# si vos paramètres w et b, correspondant à chaque classe, sont stockés sur les lignes de thetaRL... Alors:

pRL  = np.array([[1./(1+np.exp(-x.dot(mod[0]) - mod[1])) for x in X] for mod in theta ])
pRLT = np.array([[1./(1+np.exp(-x.dot(mod[0]) - mod[1])) for x in XT] for mod in theta ])
ypred  = pRL.argmax(0)
ypredT = pRLT.argmax(0)
print "Taux bonne classification en apprentissage : ",np.where(ypred != Y, 0.,1.).mean()
print "Taux bonne classification en test : ",np.where(ypredT != YT, 0.,1.).mean()