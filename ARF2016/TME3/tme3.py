# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 11:04:38 2017

@author: 3200183
"""
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def f_cos (x) :  #definition de la fonction
    return  x*np.cos(x)
#np.cos(x) - x*np.sin(x)
def f_cos_grad (x) :  #definition du gradient
    return np.cos(x) - x*np.sin(x)
    
def f_lg (x) :  #definition de la fonction -logx + x*x
    return - np.log(x)+x*x

def f_lg_grad (x) :  #definition du gradient
    return -1/x + 2*x 


def make_grid(xmin=-5,xmax=5,ymin=-5,ymax=5,step=20,data=None):
    """ Cree une grille sous forme de matrice 2d de la liste des points
    :return: une matrice 2d contenant les points de la grille, la liste x, la liste y
    """
    if data is not None:
        xmax,xmin,ymax,ymin = np.max(data[:,0]),np.min(data[:,0]),\
                              np.max(data[:,1]),np.min(data[:,1])
    x,y = np.meshgrid(np.arange(xmin,xmax,(xmax-xmin)*1./step),
                      np.arange(ymin,ymax,(ymax-ymin)*1./step))
    grid=np.c_[x.ravel(),y.ravel()]
    return grid, x, y

def load_usps(filename):
    with open(filename,"r") as f:
        f.readline()
        data =[ [float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp = np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)
    
def optimize( fonc,grad, dim , eps=0.01, maxiter=100, xinit=None ):
    
    log_x= np.ones((maxiter,dim ))
    log_f = np.ones((maxiter,1))
    log_grad= np.ones((maxiter,dim))
    
    if xinit == None :
        xinit=np.random.rand(dim)
        if len(xinit.shape)==1:
            xinit.reshape ((1 ,xinit.size)) 
    log_x[0]=xinit
    log_grad[0]=grad(xinit)
    log_f[0]=fonc(xinit)
    
    for i in np.arange(1,maxiter):
        
        xinit += -eps*grad(xinit)
        log_x[i] = xinit
        log_f[i] = fonc(xinit)
        log_grad[i] = grad(xinit)
        
    return (log_x,log_f,log_grad)

# Test avec les deux fonctions (x_cosx, -logx+x*x)

cos_x, cos_f, cos_grad = optimize(f_cos,f_cos_grad,1 ,0.01,300,1)     
lg_x, lg_f, lg_grad = optimize(f_lg,f_lg_grad,1,0.01,300,1)
"""
X=np.linspace(1,7,1000)
plt.figure()
plt.plot(X,f_cos(X),color="green") #courbe verte
plt.plot(cos_x,cos_f ,"+",color="red")
plt.show() 
"""


def f_rosenbrock(x):
    
        
        return 100*(x[:,1]-x[:,0]*x[:,0])**2+(1-x[:,0])**2
    
def f_rosenbrock_grad(x):
    grad_x1=100*(-4*x[:,1]*x[:,0]+4*x[:,0]**3)+(-2*x[:,0]+2*x[:,0])
    grad_x2=2*x[:,1]-2*x[:,0]**2
    return np.array([grad_x1,grad_x2])
    
(rosenbrock_x, rosenbrock_f, rosenbrock_grad) = optimize(f_rosenbrock,f_rosenbrock_grad,2, 0.0001, 100, np.array([3,3]).reshape(1,2)) 
 
def mafonction(data):
    liste=[]
    for x in data:
        liste.append(f_rosenbrock(x))
    return np.array(liste)    
  


 
grid,xx,yy = make_grid(-1,3,-1,3,20)
#plt.contourf(xx,yy,mafonction(grid).reshape(xx.shape),levels=255)



fig = plt.figure()

ax = fig.gca(projection='3d')
surf = ax.plot_surface(xx, yy, mafonction(grid).reshape(xx.shape),rstride=1,cstride=1,\
	  cmap=cm.gist_rainbow, linewidth=0, antialiased=False)
fig.colorbar(surf)
ax.plot(rosenbrock_x[:,0], rosenbrock_x[:,1],  rosenbrock_f.ravel(), color='black')
plt.show()

