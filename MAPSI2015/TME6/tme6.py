# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 10:41:06 2015

@author: 3200386
"""

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

data = pkl.load(file("lettres.pkl","rb"))
X = np.array(data.get('letters')) # récupération des données sur les lettres
Y = np.array(data.get('labels')) # récupération des étiquettes associées 
print Y

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


#print tracerLettre(X[0])


""" Apprentissage d'un modèle CM (max de vraisemblance) """

""" 1. Discrétisation """

def discretise(X,d):
    intervalle = 360.0/d
    discr = []
    for x in X:
        discr.append(np.floor(x/intervalle))
    return np.array(discr)

#print discretise(X, 3)

""" 2. Regrouper les indices des signaux par classe """

def groupByLabel( y):
    index = []
    for i in np.unique(y): # pour toutes les classes
        ind, = np.where(y==i)
        index.append(ind)
    return index

#print groupByLabel(Y)

""" 3. Apprendre les modèles CM """
def learnMarkovModel(Xc, d):
    a = np.zeros((d,d))
    Pi= np.zeros(d)
    for x in Xc:
        Pi[int(x[0])] += 1
        for i in np.arange(0,len(x)-1):
            a[int(x[i]),int(x[i+1])] += 1
            
    a = a/np.maximum(a.sum(1).reshape(d,1),1) # normalisation
    Pi = Pi/Pi.sum()
    return Pi,a 

def learnMarkovModelOnes(Xc, d):
    a = np.ones((d,d))
    Pi= np.ones(d)
    for x in Xc:
        Pi[int(x[0])] += 1
        for i in np.arange(0,len(x)-1):
            a[int(x[i]),int(x[i+1])] += 1
            
    a = a/np.maximum(a.sum(1).reshape(d,1),1) # normalisation
    Pi = Pi/Pi.sum()
    return Pi,a 
    
d=3     # paramètre de discrétisation
Xd = discretise(X,d)  # application de la discrétisation 
index = groupByLabel(Y)  # groupement des signaux par classe
#print learnMarkovModel(Xd[index[0]], d)  


""" 4. Stocker les modèles dans une liste  """

#d=20     # paramètre de discrétisation
Xd = discretise(X,d)  # application de la discrétisation
index = groupByLabel(Y)  # groupement des signaux par classe

models = []
models2 = []
for cl in range(len(np.unique(Y))): # parcours de toutes les classes et optimisation des modèles
    models.append(learnMarkovModel(Xd[index[cl]], d))
    
""" avec initialisation a 1 """
for cl in range(len(np.unique(Y))): # parcours de toutes les classes et optimisation des modèles
    models2.append(learnMarkovModelOnes(Xd[index[cl]], d))    
    
#print models
#print models2    
    


""" Test (affectation dans les classes sur critère MV) """

""" 1. (log)Probabilité d'une séquence dans un modèle """



def probaSequence (s,Pi, A):
    p =  np.log(Pi[int(s[0])])
    for i in np.arange (0,len(s)-1):
        p += np.log (A[int(s[i]),int(s[i+1])])
    return p

""" 2. Application de la méthode précédente pour tous les signaux et tous les modèles de lettres """

#print "normal" 
proba = np.array([[probaSequence(Xd[i], models[cl][0], models[cl][1]) for i in range(len(Xd))]for cl in range(len(np.unique(Y)))])
#proba = np.array([[probaSequence(Xd[0], models[cl][0], models[cl][1])]for cl in range(len(np.unique(Y)))])


#proba2 = np.array([[probaSequence(Xd[i], models2[cl][0], models2[cl][1]) for i in range(len(Xd))]for cl in range(len(np.unique(Y)))])
#print proba2
#print proba

""" 3. Evaluation des performances """

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


ia = []
for i in itrain:
    ia += i.tolist()    
it = []
for i in itest:
    it += i.tolist()

print it

""" itrain """
print "itrain"    
proba = np.array([[probaSequence(Xd[i], models[cl][0], models[cl][1]) for i in ia]for cl in range(len(np.unique(Y)))])
    
Ynum = np.zeros(Y.shape)
for num,char in enumerate(np.unique(Y)):
    Ynum[Y==char] = num
pred = proba.argmax(0) # max colonne par colonne
print np.where(pred != Ynum[ia], 0.,1.).mean()

""" itest """
print "itest"
proba = np.array([[probaSequence(Xd[i], models[cl][0], models[cl][1]) for i in it]for cl in range(len(np.unique(Y)))])   
Ynum = np.zeros(Y.shape)
for num,char in enumerate(np.unique(Y)):
    Ynum[Y==char] = num
pred = proba.argmax(0) # max colonne par colonne
print np.where(pred != Ynum[it], 0.,1.).mean()

print pred

""" Partie Optionnelle """


liste = []
for i in range(len(Y[it[:]])):
    liste.append(ord(Y[it[i]])-97)

print liste



conf = np.zeros((26,26))

for i,j in zip(pred,liste):
    print i
    print j
    conf[i][j] += 1


plt.figure()
plt.imshow(conf, interpolation='nearest')
plt.colorbar()
plt.xticks(np.arange(26),np.unique(Y))
plt.yticks(np.arange(26),np.unique(Y))
plt.xlabel(u'Vérité terrain')
plt.ylabel(u'Prédiction')
plt.savefig("mat_conf_lettres.png")

