# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 10:52:15 2016

@author: 3200183
"""

import numpy as np 
import matplotlib.pyplot as plt
import math as m 
from mpl_toolkits.mplot3d import Axes3D
import pickle as pkl 

fname = "dataVelib.pkl"
f= open(fname,'rb')
data = pkl.load(f)
f.close()


fin = np.zeros((0,6))
#print fin
for stations in data:
    arr = stations['number']/1000    
    if(arr>=1 and arr<=20):
        st = np.array([stations['position']['lat'],stations['position']['lng'],stations['alt'],arr,stations['bike_stands'],stations['available_bike_stands']])
        fin =np.vstack((fin,st))

#print fin 






parr = np.zeros((20),int)
#print parr
j=0
for i in fin:
    #print i[3]
    parr[i[3]-1]+=1
    j += 1 

proba_arr= parr/j
#print proba_arr
#print sum(proba_arr)


palt = np.array(fin[:,2])

"""
print "paalt"
print palt
palt= np.sort(palt)
print palt  

palt= np.array_split(palt,100)
print palt

for i in palt:
    for j in i:
 """       
        




nIntervalles = 100
res = plt.hist(palt, nIntervalles)
print res[0] # effectif dans les intervalles
print res[1] # definition des intervalles (ATTENTION: 31 valeurs)

#proba_alt = res[0]/len(fin)
#print proba_alt

#proba_sp_cond_al = 

# stocker les altitudes + définir la largeur des intervalles:
alt = res[1]
intervalle = alt[1]-alt[0]
# définir la fréquence de répartition de la population dans les intervalles
pAlt = res[0]/res[0].sum()
# puis diviser la fréquence par la base de l'intervalle pour que ça somme à 1
pAlt /= intervalle
plt.figure() # créer une figure

plt.bar((alt[1:]+alt[:-1])/2,pAlt, alt[1]-alt[0])
# NB: dans bar, on donne: abscisse, ordonnées, largeur des barres

print "fin"
print fin[:,3].astype(int)+1



#print data[]['position']
#print stations
x1 = fin[:,0] # recuperation des coordonnées
print "x1"
print x1
x2 = fin[:,1]
print x2
# définition de tous les styles (pour distinguer les arrondissements)
style = [(s,c) for s in "o^D*" for c in "byrmck" ]

print  style

# tracé de la figure
plt.figure()
for i in range(1,21):
    ind, = np.where(fin[:,3].astype(int)+1==i)
    print "ind"
    print ind
    print x1[ind]
    print x2[ind]
    # scatter c'est plus joli pour ce type d'affichage
    plt.scatter(x2[ind],x1[ind],marker=style[i-1][0],s=10,c=style[i-1][1],linewidths=0)

plt.axis('equal') # astuce pour que les axes aient les mêmes espacements
plt.legend(range(1,21), fontsize=7)
plt.savefig("carteArrondissements.pdf")
