# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 11:15:36 2016

@author: 3200183
"""

import numpy as np
import math as mat
import matplotlib.pyplot as plt
im=plt.imread ("fichier.png") [ : , : , : 3 ] #on  garde  que  les  3  premieres  composantes ,  latransparence  est  i n u t i l e

#im=plt.imread ("LosAngelesLakers.png") [ : , : , : 3 ]
im_h , im_l , _=im.shape
pixels=im . reshape (( im_h*im_l ,3) ) #transformation  en  matrice n*3 , n nombre de  pixels
imnew=pixels . reshape (( im_h , im_l ,3) ) #transformation  inverse
plt.imshow(im) #a f f i ch e r  l'image
#print pixels


def k_means(pixels,k,eps):
    past = np.ones(3)
    bary= np.array(pixels[np.random.randint(0,len(pixels),k)]) 
    pixels_clusters= np.zeros([len(pixels),3])
    while True:
        for idx,val in enumerate(pixels):
            dist =np.array([np.sqrt(np.square(j[0]-val[0])+np.square(j[1]-val[1])+np.square(j[2]-val[2])) for j in bary])
            #print np.argmin(dist)
            pixels_clusters[idx] = bary[np.argmin(dist)]
        for idx2, val2 in enumerate(bary):
            cluster_pixels = np.array([pixels[i] for i in id(pixels_clusters) if pixels_clusters == val2 ])
        pres = np.mean(cluster_pixels,axis= 0)
        res = past-pres
        if all(res<eps):
            return pixels_clusters
    
print k_means(pixels,2)



