# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 10:42:06 2015

@author: 3200386
"""

import numpy as np
from math import *
import matplotlib.pyplot as plt
import pickle as pkl
import sklearn.linear_model as lin
import sklearn.naive_bayes as nb
import sklearn.cross_validation as cv

data = pkl.load(file("/tmp/imagesRedR.pkl","rb"))

# données brutes issues de la référence de l'intro
#    chaque champ contient les caractéristiques de 100 images:
#    dans chaque champ, on peut acceder à l'image i en faisant par exemple data['im'][i]
images                   = data['im'] # 100 im : 256x256 pixels x RGB
label_pixel              = data['lab'] # 100 im : 256x256 pixels

# pour le TME, nous travaillerons sur des segments:
segments                = data['seg'] # 100 im : 256x256 pixels => 16x16 segments
label_segments          = data['segLab'] # 100 im : 256 segments x labels
coord_segments          = data['coord'] # 100 im : 256 segments x 4 coordonnées dans l'image
description_segments    = data['feat'] # 100 im: 256 segments x 9 descripteurs RGB
# pour plus de détail sur les descripteurs, voir les questions suivantes sur le modèle vectoriel
graphes                 = data['graph'] # 100 im: 256 segments x connexions hbgd
labels                 = data['labMeaning'] # signification des étiquettes
#[u'window', u'boat', u'bridge', u'building', u'mountain', u'person', u'plant', u'river', u'road', u'rock', u'sand', u'sea', u'sky', u'sun', u'tree']

##################################################################################
# Fonctions d'affichage

def drawImageTreatement(index, data):
    #plt.figure()
    fig, ax = plt.subplots(2,2)
    ax[0,0].imshow(data['im'][index], interpolation='nearest')
    ax[0,0].set_title('image de base')
    ax[0,1].imshow(data['lab'][index],interpolation='nearest')
    ax[0,1].set_title('etiquetage pixel')
    #ax[0,2].imshow(data['seg'][index],interpolation='nearest')
    ax[1,0].imshow(mark_boundaries(data['im'][index], data['seg'][index]),interpolation='nearest')
    ax[1,0].set_title('segmentation 16x16')
    newIm = data['seg'][index].copy()
    for i,reg in enumerate(np.sort(np.unique(data['seg'][index]))):
        newIm[data['seg'][index] == reg] = data['segLab'][index][i]
    ax[1,1].imshow(newIm,interpolation='nearest')
    ax[1,1].set_title('etiquetage segment')
    #plt.savefig("traitementIm.png")
    fig,ax=plt.subplots()
    color = ['blue','red','green','black']
    ax.imshow(mark_boundaries(data['im'][index], data['seg'][index]),interpolation='nearest')
    scale = len(data['im'][index])
    ax.scatter(data['coord'][index][:,2]*scale, data['coord'][index][:,0]*scale)
    for i,node in enumerate(data['graph'][index]):
        for c,side in enumerate(node):
            if c==1 or c ==3:
                continue
            #print side
            if not side == -1:
                ax.add_line(lines.Line2D(data['coord'][index][[i,side],2]*scale, data['coord'][index][[i,side],0]*scale, linewidth=1, color=color[c]))
    ax.set_title('graphe de l\'image')
    #plt.savefig("traitementImGr.png")

def introspectionModel(classif, data, Y, filename=None):
    # introspection: qu'est ce qui est associé à chaque classe de données
    plt.figure()
    plt.imshow(classif.coef_, interpolation='nearest')
    localLabs = [data['labMeaning'][y-1] for y in np.unique(Y)]
    plt.yticks(range(len(localLabs)),localLabs)
    rgb = [color+' '+strength for color in ['red', 'green', 'blue'] for strength in ['(low)', '(med)', '(high)']]
    plt.xticks(np.arange(len(rgb))-0.5,rgb, rotation=45)
    plt.vlines(np.array([2.5,5.5]), -1, len(localLabs),linestyles='dashed')
    if filename != None:
        plt.savefig(filename)

def plotTransMatrix(A, labStr):
    title = ['g','d','h','b']
    fig, ax = plt.subplots(2,2)
    c=0
    for i in range(2):
        for j in range(2):
            ax[i,j].imshow(A[:,:,c], interpolation='nearest')
            ax[i,j].set_title(title[c])
            plt.setp(ax[i,j], xticks=np.arange(len(labStr)), xticklabels=labStr,  yticks=np.arange(len(labStr)),yticklabels=labStr)
            for tick in ax[i,j].get_xticklabels():
                tick.set_rotation(90)
            c+=1
            

features  = data['feat'][1][2] # => 9 valeurs
# array([ 0.98046875,0.01953125,0. ,0. ,1. , 0. , 0. , 0. ,1.], dtype=float16)
# pas de rouge, vert à un niveau moyen et bcp de bleu => turquoise

# X : 1 segment par ligne
X = np.array([ x for i in range(len(data['feat'])) for x in data['feat'][i]])

# Y : l'étiquette correspondante
Y = np.array([ x for i in range(len(data['segLab'])) for x in data['segLab'][i]])


classifier  = lin.LogisticRegression();
classifier2 = nb.MultinomialNB(); # pas propre sur des données continues... Mais efficace

# apprentissage
classifier.fit(X,Y) # automatiquement multi-classes un-contre-tous
classifier2.fit(X,Y)

"""
# inférence sur 1 ou plusieurs vecteurs
print classifier.predict(X[0]) # rend une classe
print classifier.predict_proba(X[0]) # rend le vecteur de proba d'appartenance aux classes pour X[0]
print classifier.predict_log_proba(X[0]) # log proba

# définition d'un objet Validation Croisée:
cvk = cv.StratifiedKFold(Y, n_folds=5) # 5 sous-ensembles de données
classifier  = lin.LogisticRegression();
k=0
for train_index, test_index in cvk: # parcours des 5 sous-ensembles
    classifier.fit(X[train_index],Y[train_index])
    ypredL = classifier.predict(X[train_index])
    ypredT = classifier.predict(X[test_index])
    print "(RL) iteration ",k," pc good (Learn) ",np.where(ypredL == Y[train_index],1.,0.).sum()/len(train_index)
    print "(RL) iteration ",k," pc good (Test)  ",np.where(ypredT == Y[test_index],1.,0.).sum()/len(test_index)
    k+=1
"""

#drawImageTreatement(1, data)

#introspectionModel(classifier, data,Y)

#plotTransMatrix(A, labStr)

def MRFLearn(data):
    
    taille = len(data['labMeaning'])
    
    Ag = np.zeros((taille,taille))
    Ad = np.zeros((taille,taille))
    Ah = np.zeros((taille,taille)) 
    Ab = np.zeros((taille,taille))    
    print len(Ag)
    
    for i in range(len(data['graph'])):
        for j in range(len(data['graph'][i])):
            segG=data['graph'][i][j][0]
            segD=data['graph'][i][j][1]
            segH=data['graph'][i][j][2]
            segB=data['graph'][i][j][3]
            
            if (segG != -1):
                Ag[data['segLab'][i][j]][data['segLab'][i][segG]]+=1        
            if (segD != -1):
                Ad[data['segLab'][i][j]][data['segLab'][i][segD]]+=1        
            if (segH != -1):
                Ah[data['segLab'][i][j]][data['segLab'][i][segH]]+=1        
            if (segB != -1):
                Ab[data['segLab'][i][j]][data['segLab'][i][segB]]+=1
                
    Ag = Ag/np.maximum(Ag.sum(1).reshape(taille,1),1)            
    Ad = Ad/np.maximum(Ad.sum(1).reshape(taille,1),1)            
    Ah = Ah/np.maximum(Ah.sum(1).reshape(taille,1),1)            
    Ab = Ab/np.maximum(Ab.sum(1).reshape(taille,1),1)            
    
    return (Ag,Ad,Ah,Ab) 

    
A = MRFLearn(data)    
print A





