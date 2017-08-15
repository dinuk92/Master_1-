# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 10:36:09 2015

@author: gonzales
"""

import math
import numpy as np


def __nb_vars ( CPT ):
    """
    Calcule le nombre de variables d'une table de proba CPT
    """
    i = len ( CPT )
    nb = 0
    while i > 1:
        i /= 2
        nb += 1
    return nb
    
    
def __check_CPT ( P ):
    """
    Retourne un booleen indiquant si la probabilité P a un format correct
    """
    variables,cpt = P
    return len ( variables ) == __nb_vars ( cpt )


def marginalize1Var ( P, var ):
    """
    supprime 1 variable var d'une probabilité P par sommation

    Param P : une distribution de proba jointe sous forme d'un couple
       (liste de variables (représentées par des entiers: 0 = X_0, 1 = X_1, 
       etc.), array à 1 dimension représentant les valeurs de la 
       distribution de probabilité (toutes les variables aléatoires sont 
       supposées binaires)
    Param var : représente la variable aléatoire à marginaliser
       (0 = X_0, 1 = X_1, etc).
    """
    variables,cpt = P
    
    # recupere l'index de la variable a eliminer et cree la nouvelle
    # liste de variables
    try:
        index = variables.index ( var )
    except ValueError:
        raise ValueError ( "Marginalize: la distribution de probabilité " + 
            "ne contient pas la variable X_" + str(var) )
    variables_result = list( variables )
    variables_result.pop ( index )
    
    length = 2**( index + 1 )
    reste = 2**index
    cpt_result = np.zeros ( len ( cpt ) / 2 )
    for i in range ( len ( cpt ) ):
        j = math.floor ( i / length ) * length / 2 + ( i % reste )
        cpt_result[j] += cpt[i]
    return (variables_result,cpt_result)


def marginalize_joint_proba ( P, vars_to_remove ):
    """
    Calcul de la marginalisation d'une distribution de probas

    Param P : une distribution de proba jointe sous forme d'un couple
       (liste de variables (représentées par des entiers: 0 = X_0, 1 = X_1, 
       etc.), array à 1 dimension représentant les valeurs de la 
       distribution de probabilité (toutes les variables aléatoires sont 
       supposées binaires)
    Param vars_to_remove un array d'index representant les variables à
    supprimer: 0 = X_0, 1 = X_1, etc.
    """
    # si on ne supprime aucune variable, on retourne P
    if not vars_to_remove:
        return P
    P_result = P

    # on checke que la proba est jointe
    if abs ( 1 - np.sum ( P[1] ) ) > 1e-4:
        raise ValueError ( "marginalize_joint_probas: P ne semble pas être " +
        "une probailité jointe: la somme de ses éléments n'est pas égale à 1")

    for var in vars_to_remove:
        P_result = marginalize1Var ( P_result, var )
    return P_result
    
    
def expanse1Var ( P, var ):
    """
    Duplique une distribution de proba |var| = 2 fois. La duplication se fait
    à l'index de la variable passé en argument (les variables sont ordonnées
    par ordre croissant).
    Par exemple, si P = [0,1,2,3] et var = 0, expanse1Var renverra
    [0,0,1,1,2,2,3,3]. Si index = 1, expanse1Var renverra [0,1,0,1,2,3,2,3].

    Param P : une distribution de proba jointe sous forme d'un couple
       (liste de variables (représentées par des entiers: 0 = X_0, 1 = X_1, 
       etc.), array à 1 dimension représentant les valeurs de la 
       distribution de probabilité (toutes les variables aléatoires sont 
       supposées binaires)
    Param var : représente la variable à dupliquer (0 = X_0, 1 = X_1, etc).
    """
    variables,cpt = P
    if var in variables:
        raise ValueError ( "Expanse: la distribution de probabilité " + 
            "contient déjà la variable X_" + str(var) )

    # calcul de l'index ou devrait se trouver la variable var
    variables_result = list( variables )
    variables_result.append ( var )
    variables_result.sort ()
    index = variables_result.index ( var )
     
    length = 2**(index+1)
    reste = 2**index
    cpt_result = np.zeros ( cpt.size * 2 )
    for i in range ( cpt_result.size ):
        j = math.floor ( i / length ) * length / 2 + ( i % reste )
        cpt_result[i] = cpt[j]
    return (variables_result,cpt_result)


def expanse ( P, vars_to_add ):
    """
    Expansion d'une probabilité projetée

    Param P : une distribution de proba jointe sous forme d'un couple
       (liste de variables (représentées par des entiers: 0 = X_0, 1 = X_1, 
       etc.), array à 1 dimension représentant les valeurs de la 
       distribution de probabilité (toutes les variables aléatoires sont 
       supposées binaires)
    Param vars_to_add un array d'index representant les variables permettant
    de dupliquer la proba P. 0 = X_0, 1 = X_1, etc.
    """
    # is l'on n'a rien à expanser, on retourne P
    if not vars_to_add :
        return P
    P_result = P
    for var in vars_to_add:
        P_result = expanse1Var ( P_result, var )
    return P_result
    

def divide_joint_probas ( P1, P2 ):
    """
    Renvoie la proba jointe P1 divisee par la proba jointe P2

    Param P1 : une distribution de proba jointe sous forme d'un couple
       (liste de variables (représentées par des entiers: 0 = X_0, 1 = X_1, 
       etc.), array à 1 dimension représentant les valeurs de la 
       distribution de probabilité (toutes les variables aléatoires sont 
       supposées binaires)
    Param P2 : une distribution de proba jointe sous forme d'un couple
       (liste de variables (représentées par des entiers: 0 = X_0, 1 = X_1, 
       etc.), array à 1 dimension représentant les valeurs de la 
       distribution de probabilité (toutes les variables aléatoires sont 
       supposées binaires)
    """
    v1,cpt1 = P1
    v2,cpt2 = P2
    
    # on checke que les deux probas sont jointes
    if abs ( 1 - np.sum ( cpt1 ) ) > 1e-4:
        raise ValueError ( "divide_joint_probas: P1 ne semble pas être une " +
        "probailité jointe: la somme de ses éléments n'est pas égale à 1")
    if abs ( 1 - np.sum ( cpt2 ) ) > 1e-4:
        raise ValueError ( "divide_joint_probas: P2 ne semble pas être une " +
        "probailité jointe: la somme de ses éléments n'est pas égale à 1")
   
    # on calcule les variables qu'il faut rajouter à v1 et v2 pour
    # obtenir leur union
    v1_add = filter(lambda x: x not in v1, v2)
    v2_add = filter(lambda x: x not in v2, v1)
    new_var1,new_cpt1 = expanse ( P1, v1_add )
    new_var2,new_cpt2 = expanse ( P2, v2_add )
    
    # quand le dénominateur de la division terme à terme est égal à 0,
    # le numérateur doit également être égal à 0. Le résultat de la
    # dvision 0/0 est alors égal à 0
    index_0_P1 = np.where ( new_cpt1 == 0 )[0]
    index_0_P2 = np.where ( new_cpt2 == 0 )[0]
    for i in index_0_P2:
        if i not in index_0_P1:
            raise ValueError ( "divide_joint_probas: vous divisez un " +
            "terme non nul de P1 par un terme de P2 égal à 0" )
    
    new_cpt3 = np.where ( new_cpt2 != 0, new_cpt2, 1 ) # 0 / 0 = 0
    return ( new_var1, new_cpt1 / new_cpt3 )
    
    
def multiply_probas ( P1, P2 ):
    """
    Renvoie le produit terme à terme des deux probabilités P1 et P2

    Param P1 : une distribution de proba jointe sous forme d'un couple
       (liste de variables (représentées par des entiers: 0 = X_0, 1 = X_1, 
       etc.), array à 1 dimension représentant les valeurs de la 
       distribution de probabilité (toutes les variables aléatoires sont 
       supposées binaires)
    Param P2 : une distribution de proba jointe sous forme d'un couple
       (liste de variables (représentées par des entiers: 0 = X_0, 1 = X_1, 
       etc.), array à 1 dimension représentant les valeurs de la 
       distribution de probabilité (toutes les variables aléatoires sont 
       supposées binaires)
    """
    v1,cpt1 = P1
    v2,cpt2 = P2
    
    # on calcule les variables qu'il faut rajouter à v1 et v2 pour
    # obtenir leur union
    v1_add = filter(lambda x: x not in v1, v2)
    v2_add = filter(lambda x: x not in v2, v1)
    new_var1,new_cpt1 = expanse ( P1, v1_add )
    new_var2,new_cpt2 = expanse ( P2, v2_add )
    
    return ( new_var1, new_cpt1 * new_cpt2 )
    


def equal_conditional_probas ( P1, P2, epsilon=0.01 ):
    """
    Renvoie un booléen indiquant si deux probas sont égales

    Param P1 : une distribution de proba sous forme d'un couple
       (liste de variables (représentées par des entiers: 0 = X_0, 1 = X_1, 
       etc.), array à 1 dimension représentant les valeurs de la 
       distribution de probabilité (toutes les variables aléatoires sont 
       supposées binaires)
    Param P2 : une distribution de proba sous forme d'un couple
       (liste de variables (représentées par des entiers: 0 = X_0, 1 = X_1, 
       etc.), array à 1 dimension représentant les valeurs de la 
       distribution de probabilité (toutes les variables aléatoires sont 
       supposées binaires)
    """
    v1,cpt1 = P1
    v2,cpt2 = P2
    v1_add = filter(lambda x: x not in v1, v2)
    v2_add = filter(lambda x: x not in v2, v1)
    new_var1,new_cpt1 = expanse ( P1, v1_add )
    new_var2,new_cpt2 = expanse ( P2, v2_add )
    new_cpt3 = np.where ( new_cpt2 != 0, new_cpt2, 1 )
    incorrect = np.where ( np.abs ( new_cpt1 - new_cpt2 ) / new_cpt3 >= epsilon)
    return ( len ( incorrect[0] ) == 0 )


def read_file ( filename ):
    """
    Renvoie la probabilité contenue dans le fichier dont le nom
    est passé en argument.
    
    La distribution de proba retournée est sous la forme d'un couple
    (liste de variables (représentées par des entiers: 0 = X_0, 1 = X_1, 
    etc.), array à 1 dimension représentant les valeurs de la 
    distribution de probabilité (toutes les variables aléatoires sont 
    supposées binaires)
    """
    cpt = np.loadtxt ( filename )
    v = range ( __nb_vars ( cpt ) )
    return (v,cpt)
    
    
