# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 13:56:15 2015

@author: 3200386
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_file ( filename ):
    """
    Lit un fichier USPS et renvoie un tableau de tableaux d'images.
    Chaque image est un tableau de nombres réels.
    Chaque tableau d'images contient des images de la même classe.
    Ainsi, T = read_file ( "fichier" ) est tel que T[0] est le tableau
    des images de la classe 0, T[1] contient celui des images de la classe 1,
    et ainsi de suite.
    """
    # lecture de l'en-tête
    infile = open ( filename, "r" )    
    nb_classes, nb_features = [ int( x ) for x in infile.readline().split() ]

    # creation de la structure de données pour sauver les images :
    # c'est un tableau de listes (1 par classe)
    data = np.empty ( 10, dtype=object )  
    filler = np.frompyfunc(lambda x: list(), 1, 1)
    filler( data, data )

    # lecture des images du fichier et tri, classe par classe
    for ligne in infile:
        champs = ligne.split ()
        if len ( champs ) == nb_features + 1:
            classe = int ( champs.pop ( 0 ) )
            data[classe].append ( map ( lambda x: float(x), champs ) )    
    infile.close ()

    # transformation des list en array
    output  = np.empty ( 10, dtype=object )
    filler2 = np.frompyfunc(lambda x: np.asarray (x), 1, 1)
    filler2 ( data, output )

    return output

def display_image ( X ):
    """
    Etant donné un tableau X de 256 flotants représentant une image de 16x16
    pixels, la fonction affiche cette image dans une fenêtre.
    """
    # on teste que le tableau contient bien 256 valeurs
    if X.size != 256:
        raise ValueError ( "Les images doivent être de 16x16 pixels" )

    # on crée une image pour imshow: chaque pixel est un tableau à 3 valeurs
    # (1 pour chaque canal R,G,B). Ces valeurs sont entre 0 et 1
    Y = X / X.max ()
    img = np.zeros ( ( Y.size, 3 ) )
    for i in range ( 3 ):
        img[:,i] = X

    # on indique que toutes les images sont de 16x16 pixels
    img.shape = (16,16,3)

    # affichage de l'image
    plt.imshow( img )
    plt.show ()

training_data = read_file ( "2015_tme3_usps_train.txt" ) 

""" 2. Maximum de vraisemblance pour une classe """
   
   

def learnML_class_parameters(tab):
    
    
    nbImage = len(tab)       
    nbPixel = len(tab[0])    
    
    #print nb_imgdef
    #print nb_pix
    
    esp = np.zeros(nbPixel)
    espCarre = np.zeros(nbPixel)
    var = np.zeros(nbPixel)
    
    for i in range(nbImage):
        for j in range(nbPixel):
            esp[j] = esp[j] + tab[i][j]                           #on calcule la somme des cases de même indice pour chacune des images
            espCarre[j] = espCarre[j] + np.square(tab[i][j])      #pareil au carré
        
    for i in range(nbPixel):                                       
        esp[i] = esp[i] / nbImage                                  
        espCarre[i] = espCarre[i] / nbImage
        var[i] = espCarre[i] - np.square(esp[i])                  #Formule de Koenig  Var(X) = E[(X - E[X])^2] = E[X^2] - E[X]^2
        
        
    return (esp,var)


""" 3. Maximum de vraisemblance pour toutes les classes """


def learnML_all_parameters(tab):
   
    allParam = len(tab) 

    tabCouple = []   
    
    for i in range(allParam):
        tabCouple.append(learnML_class_parameters(tab[i]))
    
      
    return tabCouple
    


""" 4. Log-vraisemblance d'une image """

parameters = learnML_all_parameters(training_data)
test_data = read_file ( "2015_tme3_usps_test.txt" ) 
def log_likelihood(tabImage,parameters):
   
    nbPixel = len(parameters[0]) #default 256
    
    logVraisemblance = 0 
    for j in range(nbPixel):
            
        var = parameters[1][j]
        if (var != 0):

            esp = parameters[0][j]
            
            somme1 = (-1.0/2.0) * np.log(2 * np.pi * var)
            somme2 = (1.0/2.0) * ((np.square(tabImage[j] - esp))/var) 
            
            logVraisemblance += somme1 - somme2
       
    return logVraisemblance






""" 5. Log-vraisemblance d'une image (bis) """
    
def log_likelihoods(tabImage,parameters):    
    
    nbImage = len(parameters)
    tabLogVraisemblance = []    
    
    for i in range(nbImage):
        tabLogVraisemblance.append(log_likelihood(tabImage,parameters[i]))
    
    
    return tabLogVraisemblance   
    

""" 6. Classification d'une image """
def classify_image(tabImage,parameters):
    
    tabLogVraisemblance = log_likelihoods(tabImage,parameters)
    max = tabLogVraisemblance[0]
    i = 1
    indice = 0
    for i in range(len(tabLogVraisemblance)):
        if (max <= tabLogVraisemblance[i]):
            max = tabLogVraisemblance[i]
            indice = i
            
    return indice        


""" Partie optionnelle """
 
""" 7. Classification de toutes les images """

def classify_all_images(test_data,parameters):
    
    tab = np.zeros((10,10),float) 
    for i in range(10):
        for j in range(len(test_data[i])):
            k = classify_image(test_data[i][j],parameters)
            tab[i][k] +=1
        for j in range(10):
            tab[i][j] /= len(test_data[i])
    
    print tab[0,0]
    print tab[2,3]
    print tab[5,3]
    return tab

""" 8. Affichage du résultat des classifications """    
   
def dessine ( classified_matrix ):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.linspace ( 0, 9, 10 )
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, classified_matrix, rstride = 1, cstride=1 )

#dessine(classify_all_images(test_data,parameters))


#print learnML_class_parameters(training_data[1])  
#print learnML_all_parameters(training_data)  
#print log_likelihood ( test_data[2][3], parameters[1] )
#print [log_likelihood(test_data[0][0],parameters[i]) for i in range (10)]
#print log_likelihoods(test_data[1][5],parameters)
#print classify_image( test_data[1][5], parameters )
#print classify_image( test_data[4][1], parameters )
#display_image (test_data[4][1])
#print classify_image( test_data[1][5], parameters )
#print classify_all_images(test_data,parameters) 