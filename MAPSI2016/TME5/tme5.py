# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 10:44:43 2016

@author: 3200183
"""

import numpy as np
import scipy.stats as stats
import pydot        
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


""" 1. Lecture des données """


# fonction pour transformer les données brutes en nombres de 0 à n-1
def translate_data ( data ):
    # création des structures de données à retourner
    nb_variables = data.shape[0]
    nb_observations = data.shape[1] - 1 # - nom variable
    res_data = np.zeros ( (nb_variables, nb_observations ), int )
    res_dico = np.empty ( nb_variables, dtype=object )

    # pour chaque variable, faire la traduction
    for i in range ( nb_variables ):
        res_dico[i] = {}
        index = 0
        for j in range ( 1, nb_observations + 1 ):
            # si l'observation n'existe pas dans le dictionnaire, la rajouter
            if data[i,j] not in res_dico[i]:
                res_dico[i].update ( { data[i,j] : index } )
                index += 1
            # rajouter la traduction dans le tableau de données à retourner
            res_data[i,j-1] = res_dico[i][data[i,j]]
    return ( res_data, res_dico )


# fonction pour lire les données de la base d'apprentissage
def read_csv ( filename ):
    data = np.loadtxt ( filename, delimiter=',', dtype='string' ).T
    names = data[:,0].copy ()
    data, dico = translate_data ( data )
    return names, data, dico

# names : tableau contenant les noms des variables aléatoires
# data  : tableau 2D contenant les instanciations des variables aléatoires
# dico  : tableau de dictionnaires contenant la correspondance (valeur de variable -> nombre)
names, data, dico = read_csv ( "2015_tme5_asia.csv" )

#print names
#print data 
#print dico 


# etant donné une BD data et son dictionnaire, cette fonction crée le
# tableau de contingence de (x,y) | z
def create_contingency_table ( data, dico, x, y, z ):
    # détermination de la taille de z
    size_z = 1
    offset_z = np.zeros ( len ( z ) )
    j = 0
    for i in z:
        offset_z[j] = size_z      
        size_z *= len ( dico[i] )
        j += 1

    # création du tableau de contingence
    res = np.zeros ( size_z, dtype = object )

    # remplissage du tableau de contingence
    if size_z != 1:
        z_values = np.apply_along_axis ( lambda val_z : val_z.dot ( offset_z ),
                                         1, data[z,:].T )
        i = 0
        while i < size_z:
            indices, = np.where ( z_values == i )
            a,b,c = np.histogram2d ( data[x,indices], data[y,indices],
                                     bins = [ len ( dico[x] ), len (dico[y] ) ] )
            res[i] = ( indices.size, a )
            i += 1
    else:
        a,b,c = np.histogram2d ( data[x,:], data[y,:],
                                 bins = [ len ( dico[x] ), len (dico[y] ) ] )
        res[0] = ( data.shape[1], a )
    return res


""" 2. Statistique du χ2 conditionnel """
 
""" 3. Statistique du χ2 et degré de liberté """


def sufficient_statistics ( data, dico, x, y, z ):
    contingence = create_contingency_table (data, dico, x, y, z )
    chi2 = 0.0
    nbZ = 0
    
    DoF = (len(dico[0])-1)*(len(dico[1])-1)   
    
    for e1 in contingence:
        Nz = e1[0]
        Txyz = e1[1]
        if(Nz != 0):
            nbZ +=1
            #print e1
            for i in range(len(Txyz[0])):
                for j in range(len(Txyz[1])):
                    Nxz = e1[1][i,:].sum()
                    Nyz = e1[1][:,j].sum()   
                    if ((Nxz*Nyz)/Nz) != 0:
                        chi2 += ((Txyz[i,j] - (Nxz*Nyz)/Nz)**2)/((Nxz*Nyz)/Nz)

    DoF *= nbZ
    return (chi2,DoF)
 
"""   
#Test    

print sufficient_statistics ( data, dico, 1,2,[3])
print sufficient_statistics ( data, dico, 0,1,[2,3])
print sufficient_statistics ( data, dico, 1,3,[2])
print sufficient_statistics ( data, dico, 5,2,[1,3,6])
print sufficient_statistics ( data, dico, 0,7,[4,5])
print sufficient_statistics ( data, dico, 2,3,[5])

"""


""" 4. Test d'indépendance """

def indep_score(data, dico, x,y,z):
    (chi2,DoF) = sufficient_statistics ( data, dico, x,y,z)
    stat = stats.chi2.sf(chi2,DoF)
    tailleX = len(dico[x])
    tailleY = len(dico[y])
    tailleZ = 0
    for i in z:
        tailleZ += len(dico[i])
    
    dmin = 5 * tailleX * tailleY * tailleZ
    if (len(data[0]) >= dmin):
        return (stat,DoF)
    return (-1,1)
    
"""
#Test
    
print indep_score ( data, dico, 1,3,[])
print indep_score ( data, dico, 1, 7, [])
print indep_score ( data, dico, 0, 1,[2, 3])
print indep_score( data, dico, 1, 2,[3, 4])
"""

""" 5. Meilleur candidat pour être un parent """

def best_candidate ( data, dico, x, z, alpha ):
     
    iy = 0
    pvalue = indep_score( data, dico, x, 0, z )
    min = pvalue[0]
    
    for y in range(1,x):
        pvalue = indep_score( data, dico, x, y, z )
        if pvalue[0] < min:
            min = pvalue[0]
            iy = y
    
            
    if min > alpha or min == -1:
        return []
    else:
        return [iy]
        
"""
#Test        
        
print best_candidate( data, dico, 4, [1], 0.05 )
print best_candidate ( data, dico, 4, [], 0.05 )
print best_candidate ( data, dico, 4, [1], 0.05 )
print best_candidate ( data, dico, 5, [], 0.05 )
print best_candidate ( data, dico, 5, [6], 0.05 )
print best_candidate ( data, dico, 5, [6,7], 0.05 )
"""

""" 6. Création des parents d'un noeud """

def  create_parents ( data, dico, x, alpha ) :
    if x == 0:
        return []
    z = np.zeros(0,int)
    y = np.zeros(1,int) 
    while   len(y) != 0:
        y = best_candidate(data, dico, x, z, alpha)
        z = np.append(z,y)
    return z


#Test
print create_parents ( data, dico, 0, 0.05 )
print create_parents ( data, dico, 1, 0.05 )
print create_parents ( data, dico, 4, 0.05 )
print create_parents ( data, dico, 5, 0.05 )
print create_parents ( data, dico, 6, 0.05 )


""" 7. Apprentissage de la structure d'un réseau bayésien """

def learn_BN_structure ( data, dico, alpha ) :
    t_res =[]
    
    for i in range(0,len(data)):
        z = create_parents(data,dico,i, alpha)
        t_res.append(z)

    return t_res
    
res = learn_BN_structure(data,dico,0.05)

print res



style = { "bgcolor" : "#6b85d1", "fgcolor" : "#FFFFFF" }

def display_BN ( node_names, bn_struct, bn_name, style ):
    graph = pydot.Dot( bn_name, graph_type='digraph')

    # création des noeuds du réseau
    for name in node_names:
        new_node = pydot.Node( name,
                               style="filled",
                               fillcolor=style["bgcolor"],
                               fontcolor=style["fgcolor"] )
        graph.add_node( new_node )

    # création des arcs
    for node in range ( len ( node_names ) ):
        parents = bn_struct[node]
        for par in parents:
            new_edge = pydot.Edge ( node_names[par], node_names[node] )
            graph.add_edge ( new_edge )

    # sauvegarde et affaichage
    outfile = bn_name + '.png'
    graph.write_png( outfile )
    img = mpimg.imread ( outfile )
    plt.imshow( img )

display_BN(names,res,"reseau",style)


