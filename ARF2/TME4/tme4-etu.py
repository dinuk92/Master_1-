#from arftools import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from arftools import *


def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def show_usps(data):
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")
    
# Cout du perceptron
def hinge(datax,datay,w):
    return np.sum(np.maximum(0,-np.dot(datay,np.dot(datax,w))))

# gradient
def hinge_grad(datax,datay,w):
     
    somme=np.zeros((len(w)))
    for i in range (0,len(datax)):
        if (np.sign(datay[i]*np.dot(datax[i],w))==-1):
            somme+=(datay[i]*datax[i])
        
    return -somme    

class Perceptron(object):
    def __init__(self,loss=hinge,loss_g=hinge_grad,max_iter=100,eps=0.01):
        self.max_iter, self.eps = max_iter,eps
        self.w = None
        self.w_histo,self.loss_histo = None,None
        self.loss = loss
        self.loss_g = loss_g

    def fit(self,datax,datay):
        datay = datay.reshape(-1,1)
        N = len(datay)
        datax = datax.reshape(N,-1)
        D = datax.shape[1]
        self.w = np.random.random((1,D))
        self.w = self.w.T
        self.w = np.array(self.w)
        self.w_histo = []
        self.loss_histo = []
        for i in range(self.max_iter):
            result = self.optimize(self.loss,self.loss_g, datax, datay)
        self.w = result[0][len(result[0])-1]
        return result
        
    def predict(self,datax):
        liste=[]
        for i in datax:
            liste.append( np.dot(np.array(self.w), np.array(i) ) )
        return np.array(liste)    
    
    def score(self,datax,datay):
        return np.mean(self.predict(datax)==datay)
    
    def optimize(self, loss, loss_g, datax, datay):
     
        log_w = list()    
        log_f = list()
        log_grad = list()    
        
        f = loss(datax, datay, self.w)
        grad = loss_g(datax, datay, self.w)
    
        log_w.append(self.w)
        log_f.append(f)
        log_grad.append(grad)   
        
        self.w = self.w - self.eps * loss_g(datax, datay, self.w)    
        
        return (log_w, log_f, log_grad)


p = Perceptron(loss=hinge,loss_g=hinge_grad, max_iter=100, eps=0.01)
#(x, y) = gen_arti(centerx=1, centery=1, sigma=0.1, nbex=1000, data_type=0, epsilon=0.02)
(x1, y1) = gen_arti(centerx=1, centery=1, sigma=0.5, nbex=100, data_type=0, epsilon=0.02)
#(x2, y2) = gen_arti(centerx=1, centery=1, sigma=0.05, nbex=100, data_type=0, epsilon=0.02)
#(log_w, log_f, log_grad) = p.fit(x1,y1)


#plot_frontiere(x2, p.predict,step=20)
#plot_data(x2, y2)
         