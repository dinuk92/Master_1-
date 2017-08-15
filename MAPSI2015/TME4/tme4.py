# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 13:39:28 2015

@author: 3200386
"""


import numpy as np
import matplotlib.pyplot as plt
import math as m
from pylab import *

def read_file ( filename ):
    """
    Lit le fichier contenant les données du geyser Old Faithful
    """
    # lecture de l'en-tête
    infile = open ( filename, "r" )
    for ligne in infile:
        if ligne.find ( "eruptions waiting" ) != -1:
            break

    # ici, on a la liste des temps d'éruption et des délais d'irruptions
    data = []
    for ligne in infile:
        nb_ligne, eruption, waiting = [ float (x) for x in ligne.split () ]
        data.append ( eruption )
        data.append ( waiting )
    infile.close ()

    # transformation de la liste en tableau 2D
    data = np.asarray ( data )
    data.shape = (data.size / 2, 2 )

    return data

data = read_file ( "2015_tme4_faithful.txt" )


""" 2. Loi normale bidimensionnelle """

def normale_bidim(x,z,tab):
   
    mux = tab[0]
    muz = tab[1]
    sigmax = tab[2]
    sigmaz = tab[3]
    rho = tab[4]
   
    expression1 = 1.0 / (2 * np.pi * sigmax * sigmaz * m.sqrt(1 - np.square(rho)))
    expression2 = -1.0 / (2 * (1 - np.square(rho)))
    expression3 = np.square((x - mux) / sigmax)
    expression4 = (-2) * rho * ((x - mux) * (z - muz)) / (sigmax * sigmaz)
    expression5 = np.square((z - muz) / sigmaz)
    
    expressionFinale = expression1 * (np.exp(expression2 * (expression3 + expression4 + expression5)))
    
    return expressionFinale


""" 3. Visualisation de loi normale bidimensionnelle """
    
def dessine_1_normale(params):
    # récupération des paramètres
    mu_x, mu_z, sigma_x, sigma_z, rho = params

    # on détermine les coordonnées des coins de la figure
    x_min = mu_x - 2 * sigma_x
    x_max = mu_x + 2 * sigma_x
    z_min = mu_z - 2 * sigma_z
    z_max = mu_z + 2 * sigma_z

    # création de la grille
    x = np.linspace ( x_min, x_max, 100 )
    z = np.linspace ( z_min, z_max, 100 )
    X, Z = np.meshgrid(x, z)

    # calcul des normales
    norm = X.copy ()
    for i in range ( x.shape[0] ):
        for j in range ( z.shape[0] ):
            norm[i,j] = normale_bidim ( x[i], z[j], params )

    # affichage
    plt.figure ()
    plt.contour ( X, Z, norm, cmap=cm.autumn )
    plt.show ()    
    

""" 4. Visualisation des données du Old Faithful """

def dessine_normales ( data, params, weights, bounds, ax ):
    # récupération des paramètres
    mu_x0, mu_z0, sigma_x0, sigma_z0, rho0 = params[0]
    mu_x1, mu_z1, sigma_x1, sigma_z1, rho1 = params[1]

    # on détermine les coordonnées des coins de la figure
    x_min = bounds[0]
    x_max = bounds[1]
    z_min = bounds[2]
    z_max = bounds[3]

    # création de la grille
    nb_x = nb_z = 100
    x = np.linspace ( x_min, x_max, nb_x )
    z = np.linspace ( z_min, z_max, nb_z )
    X, Z = np.meshgrid(x, z)

    # calcul des normales
    norm0 = np.zeros ( (nb_x,nb_z) )
    for j in range ( nb_z ):
        for i in range ( nb_x ):
            norm0[j,i] = normale_bidim ( x[i], z[j], params[0] )# * weights[0]
    norm1 = np.zeros ( (nb_x,nb_z) )
    for j in range ( nb_z ):
        for i in range ( nb_x ):
             norm1[j,i] = normale_bidim ( x[i], z[j], params[1] )# * weights[1]

    # affichages des normales et des points du dataset
    ax.contour ( X, Z, norm0, cmap=cm.winter, alpha = 0.5 )
    ax.contour ( X, Z, norm1, cmap=cm.autumn, alpha = 0.5 )
    for point in data:
        ax.plot ( point[0], point[1], 'k+' )


def find_bounds ( data, params ):
    # récupération des paramètres
    mu_x0, mu_z0, sigma_x0, sigma_z0, rho0 = params[0]
    mu_x1, mu_z1, sigma_x1, sigma_z1, rho1 = params[1]

    # calcul des coins
    x_min = min ( mu_x0 - 2 * sigma_x0, mu_x1 - 2 * sigma_x1, data[:,0].min() )
    x_max = max ( mu_x0 + 2 * sigma_x0, mu_x1 + 2 * sigma_x1, data[:,0].max() )
    z_min = min ( mu_z0 - 2 * sigma_z0, mu_z1 - 2 * sigma_z1, data[:,1].min() )
    z_max = max ( mu_z0 + 2 * sigma_z0, mu_z1 + 2 * sigma_z1, data[:,1].max() )

    return ( x_min, x_max, z_min, z_max )


# affichage des données : calcul des moyennes et variances des 2 colonnes
mean1 = data[:,0].mean ()
mean2 = data[:,1].mean ()
std1  = data[:,0].std ()
std2  = data[:,1].std ()

# les paramètres des 2 normales sont autour de ces moyennes
params = np.array ( [(mean1 - 0.2, mean2 - 1, std1, std2, 0),
                     (mean1 + 0.2, mean2 + 1, std1, std2, 0)] )
weights = np.array ( [0.4, 0.6] )
bounds = find_bounds ( data, params )

# affichage de la figure
fig = plt.figure ()
ax = fig.add_subplot(111)
dessine_normales ( data, params, weights, bounds, ax )
plt.show ()
  
""" 5. EM : l'étape E  """
    
def Q_i(data,current_param,current_weight):
    
    pi0 = current_weight[0]       
    pi1 = current_weight[1]
    tab = np.zeros((len(data),2))
    for i in range(len(data)):    
    
        alpha0 = pi0 * normale_bidim(data[i][0],data[i][1],current_param[0])
        alpha1 = pi1 * normale_bidim(data[i][0],data[i][1],current_param[1])
        tab[i][0] = (alpha0)/(alpha0 + alpha1)
        tab[i][1] = (alpha1)/(alpha0 + alpha1)
    
    return tab

""" 6. EM : l'étape M """

def M_step ( data, Q) :

    sommeQ0 = 0
    sommeQ1 = 0
    sommeMux0 = 0
    sommeMux1 = 0
    sommeMuz0 = 0
    sommeMuz1 = 0
    sommeSigmax0 = 0    
    sommeSigmax1 = 0
    sommeSigmaz0 = 0
    sommeSigmaz1 = 0
    sommeRho0 = 0
    sommeRho1 = 0
    pi0 = 0
    pi1 = 0
    for i in range(len(Q)):
        sommeQ0 += Q[i][0]
        sommeQ1 += Q[i][1]
        sommeMux0 += Q[i][0] * data[i][0]
        sommeMux1 += Q[i][1] * data[i][0]
        sommeMuz0 += Q[i][0] * data[i][1]
        sommeMuz1 += Q[i][1] * data[i][1]
        
    pi0 = sommeQ0 / (sommeQ0+sommeQ1) 
    pi1 = sommeQ1 / (sommeQ0+sommeQ1) 
    mux0 = sommeMux0 / sommeQ0
    mux1 = sommeMux1 / sommeQ1
    muz0 = sommeMuz0 / sommeQ0
    muz1 = sommeMuz1 / sommeQ1
    
    for i in range(len(Q)):    
        sommeSigmax0 += Q[i][0] * np.square(data[i][0] - mux0)
        sommeSigmax1 += Q[i][1] * np.square(data[i][0] - mux1)       
        sommeSigmaz0 += Q[i][0] * np.square(data[i][1] - muz0)
        sommeSigmaz1 += Q[i][1] * np.square(data[i][1] - muz1)
    
    sigmax0 = np.sqrt(sommeSigmax0 / sommeQ0)
    sigmax1 = np.sqrt(sommeSigmax1 / sommeQ1)
    sigmaz0 = np.sqrt(sommeSigmaz0 / sommeQ0)
    sigmaz1 = np.sqrt(sommeSigmaz1 / sommeQ1)

    for i in range(len(Q)):
        sommeRho0 += Q[i][0] * ((data[i][0] - mux0) * (data[i][1] - muz0)) / (sigmax0 * sigmaz0)  
        sommeRho1 += Q[i][1] * ((data[i][0] - mux1) * (data[i][1] - muz1)) / (sigmax1 * sigmaz1)
    
    rho0 = sommeRho0 / sommeQ0    
    rho1 = sommeRho1 / sommeQ1
    
    currParam = np.array([(mux0,muz0,sigmax0,sigmaz0,rho0),(mux1,muz1,sigmax1,sigmaz1,rho1)])
    currWeight = np.array([pi0,pi1])
    return (currParam,currWeight)

""" 7. Algorithme EM : mise au point """

def algoEM(data, params, weights, nbIteration) :
    
    param = params
    weight = weights
    
    for i in range(nbIteration):
        Q = Q_i(data,param,weight)
        (param,weight) = M_step(data,Q)
        
    return (param,weight)

# affichage des données : calcul des moyennes et variances des 2 colonnes
mean1 = data[:,0].mean ()
mean2 = data[:,1].mean ()
std1  = data[:,0].std ()
std2  = data[:,1].std ()

# les paramètres des 2 normales sont autour de ces moyennes
params = np.array ( [(mean1 - 0.2, mean2 - 1, std1, std2, 0),
                     (mean1 + 0.2, mean2 + 1, std1, std2, 0)] )
weights = np.array ( [0.5, 0.5] )
bounds = find_bounds ( data, params )

# affichage de la figure
(params,weights) = algoEM(data,params,weights,20)
fig = plt.figure ()
ax = fig.add_subplot(111)
dessine_normales ( data, params, weights, bounds, ax )
plt.show ()     
     
#print normale_bidim(1,2,[1.0,2.0,3.0,4.0,0])    
#print normale_bidim(1,0,[1.0,2.0,1.0,2.0,0.7])  
#print dessine_1_normale([-3.0,-5.0,3.0,2.0,0.7])
#print dessine_1_normale([-3.0,-5.0,3.0,2.0,0.05])
     
"""
current_params = np.array([[ 3.28778309, 69.89705882, 1.13927121, 13.56996002, 0. ],
                           [ 3.68778309, 71.89705882, 1.13927121, 13.56996002, 0. ]])
current_weights = np.array ( [ 0.5, 0.5 ] )
T = Q_i(data,current_params,current_weights)   
print T


current_params = np.array([[ 3.2194684, 67.83748075, 1.16527301, 13.9245876,  0.9070348 ],
                           [ 3.75499261, 73.9440348, 1.04650191, 12.48307362, 0.88083712]])
current_weights = np.array ( [ 0.49896815, 0.50103185] )
T = Q_i(data,current_params,current_weights)
print T


current_params = array([(2.51460515, 60.12832316, 0.90428702, 11.66108819, 0.86533355),
                        (4.2893485,  79.76680985, 0.52047055,  7.04450242, 0.58358284)])
current_weights = array([ 0.45165145,  0.54834855])
Q = Q_i ( data, current_params, current_weights )
print M_step ( data, Q)
"""


""" 8. Algorithme EM : version finale et animation """ 




# affichage des données : calcul des moyennes et variances des 2 colonnes
mean1 = data[:,0].mean ()
mean2 = data[:,1].mean ()
std1  = data[:,0].std ()
std2  = data[:,1].std ()

# les paramètres des 2 normales sont autour de ces moyennes
params = np.array ( [(mean1 - 0.2, mean2 - 1, std1, std2, 0),
                     (mean1 + 0.2, mean2 + 1, std1, std2, 0)] )
weights = np.array ( [0.5, 0.5] )
bounds = find_bounds ( data, params )


def algoEMList(data, params, weights, nbIteration) :

    param = params
    weight = weights
    liste = []
    
    for i in range(nbIteration):
        Q = Q_i(data,param,weight)
        (param,weight) = M_step(data,Q)
        liste.append((param,weight))
        
    return liste

res_EM = algoEMList(data,params,weights,20)

# calcul des bornes pour contenir toutes les lois normales calculées
def find_video_bounds ( data, res_EM ):
    bounds = np.asarray ( find_bounds ( data, res_EM[0][0] ) )
    for param in res_EM:
        new_bound = find_bounds ( data, param[0] )
        for i in [0,2]:
            bounds[i] = min ( bounds[i], new_bound[i] )
        for i in [1,3]:
            bounds[i] = max ( bounds[i], new_bound[i] )
    return bounds

bounds = find_video_bounds ( data, res_EM )


import matplotlib.animation as animation

# création de l'animation : tout d'abord on crée la figure qui sera animée
fig = plt.figure ()
ax = fig.gca (xlim=(bounds[0], bounds[1]), ylim=(bounds[2], bounds[3]))

# la fonction appelée à chaque pas de temps pour créer l'animation
def animate ( i ):
    ax.cla ()
    dessine_normales (data, res_EM[i][0], res_EM[i][1], bounds, ax)
    ax.text(5, 40, 'step = ' + str ( i ))
    print "step animate = %d" % ( i )

# exécution de l'animation
anim = animation.FuncAnimation(fig, animate,
                               frames = len ( res_EM ), interval=500 )
plt.show ()

# éventuellement, sauver l'animation dans une vidéo
anim.save('old_faithful.avi', bitrate=4000)

