# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 10:49:15 2016

@author: 3200183
"""

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

data = pkl.load(file("lettres.pkl","rb"))
X = np.array(data.get('letters')) # récupération des données sur les lettres
Y = np.array(data.get('labels')) # récupération des étiquettes associées

#print X[0]

# affichage d'une lettre
def tracerLettre(let):
    a = -let*np.pi/180; # conversion en rad
    coord = np.array([[0, 0]]); # point initial
    for i in range(len(a)):
        x = np.array([[1, 0]]);
        rot = np.array([[np.cos(a[i]), -np.sin(a[i])],[ np.sin(a[i]),np.cos(a[i])]])
        xr = x.dot(rot) # application de la rotation
        coord = np.vstack((coord,xr+coord[-1,:]))
    plt.figure()
    plt.plot(coord[:,0],coord[:,1])
    plt.savefig("exlettre.png")
    return
    
#tracerLettre(X[0])
    
""" Apprentissage d'un modèle CM (max de vraisemblance) """

""" 1. Discrétisation """

def discretise(X,d):
    intervalle = 360.0/d
    discr = np.array([np.floor(x/intervalle) for x in X])
    return discr
    
#print discretise(X,3)

"""2. Regrouper les indices des signaux par classe (pour faciliter l'apprentissage)"""

def groupByLabel( y):
    index = []
    for i in np.unique(y): # pour toutes les classes
        ind, = np.where(y==i)
        index.append(ind)
    return index

#print groupByLabel(Y)

"""3. Apprendre les modèles CM """

def learnMarkovModel(Xc, d):
    A = np.zeros((d,d))
    Pi= np.zeros(d)
    for x in Xc:
        #print int(x[0])
        #print Pi
        Pi[int(x[0])] += 1
        for i in np.arange(0,len(x)-1):
           A[int(x[i]),int(x[i+1])] += 1
            
    A = A/np.maximum(A.sum(1).reshape(d,1),1) # normalisation
    Pi = Pi/Pi.sum()
    return Pi,A 

d=3     # paramètre de discrétisation
Xd = discretise(X,d)  # application de la discrétisation
index = groupByLabel(Y)  # groupement des signaux par classe
models = []
for cl in range(len(np.unique(Y))): # parcours de toutes les classes et optimisation des modèles
    models.append(learnMarkovModel(Xd[index[cl]], d))
#print models[0]


"""4. Stocker les modèles dans une liste """

d=3     # paramètre de discrétisation
Xd = discretise(X,d)  # application de la discrétisation
index = groupByLabel(Y)  # groupement des signaux par classe
models = []
for cl in range(len(np.unique(Y))): # parcours de toutes les classes et optimisation des modèles
    models.append(learnMarkovModel(Xd[index[cl]], d))


""" Test (affectation dans les classes sur critère MV) """

""" 1. (log)Probabilité d'une séquence dans un modèle """

d=3    # paramètre de discrétisation
Xd = discretise(X,d)  # application de la discrétisation
index = groupByLabel(Y)  # groupement des signaux par classe
models = []

for cl in range(len(np.unique(Y))): # parcours de toutes les classes et optimisation des modèles
    models.append(learnMarkovModel(Xd[index[cl]], d))


def probaSequance(s,Pi,A):
    logP = np.log(Pi[s[0]])
    for j in range(len(s)-1):
        logP+= np.log(A[s[j],s[j+1]])
    return logP

proba = np.array([[probaSequance(Xd[i], models[cl][0], models[cl][1]) for i in range(1)]for cl in range(len(np.unique(Y)))])
#print proba

""" 2. Application de la méthode précédente pour tous les signaux et tous les modèles de lettres """  
    
proba = np.array([[probaSequance(Xd[i], models[cl][0], models[cl][1]) for i in range(len(Xd))]for cl in range(len(np.unique(Y)))])
   
""" 3. Evaluation des performances """
Ynum = np.zeros(Y.shape)
for num,char in enumerate(np.unique(Y)):
    Ynum[Y==char] = num

pred = proba.argmax(0) # max colonne par colonne

print np.where(pred != Ynum, 0.,1.).mean()



d=20     # paramètre de discrétisation
Xd = discretise(X,d)  # application de la discrétisation
index = groupByLabel(Y)  # groupement des signaux par classe
models = []
for cl in range(len(np.unique(Y))): # parcours de toutes les classes et optimisation des modèles
    models.append(learnMarkovModel(Xd[index[cl]], d))
    
d=20   # paramètre de discrétisation
Xd = discretise(X,d)  # application de la discrétisation
index = groupByLabel(Y)  # groupement des signaux par classe
models = []

for cl in range(len(np.unique(Y))): # parcours de toutes les classes et optimisation des modèles
    models.append(learnMarkovModel(Xd[index[cl]], d))

proba = np.array([[probaSequance(Xd[i], models[cl][0], models[cl][1]) for i in range(len(Xd))]for cl in range(len(np.unique(Y)))])
 
Ynum = np.zeros(Y.shape)
for num,char in enumerate(np.unique(Y)):
    Ynum[Y==char] = num

pred = proba.argmax(0) # max colonne par colonne

print np.where(pred != Ynum, 0.,1.).mean()  


""" Biais d'évaluation, notion de sur-apprentissage """

# separation app/test, pc=ratio de points en apprentissage
def separeTrainTest(y, pc):
    indTrain = []
    indTest = []
    for i in np.unique(y): # pour toutes les classes
        ind, = np.where(y==i)
        n = len(ind)
        indTrain.append(ind[np.random.permutation(n)][:np.floor(pc*n)])
        indTest.append(np.setdiff1d(ind, indTrain[-1]))
    return indTrain, indTest

# exemple d'utilisation
itrain,itest = separeTrainTest(Y,0.8)

#print itrain

ia = []
for i in itrain:
    ia += i.tolist()    
it = []
for i in itest:
    it += i.tolist()
    
""" itrain """
d=3     # paramètre de discrétisation
Xd = discretise(X,d)  # application de la discrétisation
index = groupByLabel(it)  # groupement des signaux par classe
models = []
for cl in range(len(np.unique(Y))): # parcours de toutes les classes et optimisation des modèles
    models.append(learnMarkovModel(Xd[itrain[cl]], d))
    
""" itest """
proba = np.array([[probaSequance(Xd[i], models[cl][0], models[cl][1]) for i in it]for cl in range(len(np.unique(Y)))])   
Ynum = np.zeros(Y.shape)
for num,char in enumerate(np.unique(Y)):
    Ynum[Y==char] = num
pred = proba.argmax(0) # max colonne par colonne
print np.where(pred != Ynum[it], 0.,1.).mean()


""" Partie optionnelle """

""" Evaluation qualitative """
liste = []
for i in range(len(Y[it[:]])):
    liste.append(ord(Y[it[i]])-97)

print liste

conf = np.zeros((26,26))

for i,j in zip(pred,liste):
    conf[i][j] += 1


plt.figure()
plt.imshow(conf, interpolation='nearest')
plt.colorbar()
plt.xticks(np.arange(26),np.unique(Y))
plt.yticks(np.arange(26),np.unique(Y))
plt.xlabel(u'Vérité terrain')
plt.ylabel(u'Prédiction')
plt.savefig("mat_conf_lettres.png")

"""Modèle génératif"""

def generate(Pi, A , n):
    sequance = []
    rd =np.random.rand()
    pic =np.cumsum(Pi)
    for p in range(len(Pi)):
        if rd <= pic[p]:
            sequance.append(p)
            break;
    acc =np.cumsum(A,1)
    for j in range(n-1):
        rd = np.random.rand()
        for i in range(len(Pi)):
            if rd <= acc[sequance[j],i]:
                sequance.append(i)
                break;
    return sequance

newa = generate(models[0][0],models[0][1], 25) # generation d'une séquence d'états
intervalle = 360./d # pour passer des états => valeur d'angles
newa_continu = np.array([i*intervalle for i in newa]) # conv int => double
tracerLettre(newa_continu)
