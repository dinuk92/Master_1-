import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from arftools import *
import random as rn

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
        if (np.sign(datay[i]*np.dot(datax[i],w))[0]==-1):
            somme+=(datay[i]*datax[i])
        
    return -somme

# le cas stockastique    
def hinge_grad_s(datax,datay,w):

    hazard=rn.randint(0,len(datax)-1)    
    while (np.sign(datay[hazard]*np.dot(datax[hazard],w))==1):
            hazard=rn.randint(0,len(datax)-1)

            
    return -(datay[hazard]*datax[hazard])
    
    
def v2m(x):
    return x.reshape ((1 ,x.size)) if len(x.shape)==1 else x
    
def v2m2(x):
    return x.reshape ((x.size,1)) if len(x.shape)==1 else x
    
class Perceptron(object):
    def __init__(self,loss=hinge,loss_g=hinge_grad,p=None,max_iter=100,eps=0.01):
        self.max_iter, self.eps = max_iter,eps
        self.w = None
        self.w_histo,self.loss_histo = None,None
        self.loss = loss
        self.p=p
        self.loss_g = loss_g

    def fit(self,datax,datay):
        #datay = datay.reshape(-1,1)
        N = len(datay)
        #datax = datax.reshape(N,-1)
        
        if (self.p== None ):
            print "test1"
            datax=np.hstack((np.ones((N,1)),datax.reshape(N,-1)))
        elif (self.p == "prog3D")  :
            print "test2"
            d3 = v2m2(np.array(datax[:,1]*datax[:,0]))
                        
            print d3.shape
            print datax.reshape(N,-1).shape
            print np.ones((N,1)).shape
            
            datax=np.hstack((datax.reshape(N,-1),d3   ))
            datax=np.hstack((np.ones((N,1)),datax.reshape(N,-1)))
            
        D = datax.shape[1]
        print D
        self.w = np.random.random((1,D))
        self.w = self.w.T
        #self.w = np.array(self.w)
        self.w_histo = []
        self.loss_histo = []
        
        for i in range(self.max_iter):
            f = self.loss(datax, datay, self.w)
            
            
            self.w_histo.append(self.w)
            self.loss_histo.append(f)
            a=v2m2(self.loss_g(datax, datay, self.w))
            
            #a=self.loss_g(datax, datay, self.w).reshape(3,1)   
            self.w = self.w - self.eps * a    
           
        return (self.w_histo, self.loss_histo)
            
    def predict(self,datax):
        N = len(datax)
        if (self.p== None ):
            print "test1"
            datax=np.hstack((np.ones((N,1)),datax.reshape(N,-1)))
        elif (self.p == "prog3D")  :
            print "test2"
            d3 = v2m2(np.array(datax[:,1]*datax[:,0]))
                        
            print d3.shape
            print datax.reshape(N,-1).shape
            print np.ones((N,1)).shape
            
            datax=np.hstack((datax.reshape(N,-1),d3   ))
            datax=np.hstack((np.ones((N,1)),datax.reshape(N,-1)))
        liste=[]
        for i in datax:            
            liste.append( np.dot(v2m(i),self.w ) )
        return np.array(liste)
        
    def score(self,datax,datay):
        return np.mean(self.predict(datax)==datay)
 #==============================================================================
# Test du Perceptron
#==============================================================================


p = Perceptron(loss=hinge,loss_g=hinge_grad, max_iter=100, eps=0.01)
# Pour le cas stockastique
#p = Perceptron(loss=hinge,loss_g=hinge_grad_s, max_iter=100, eps=0.01)
(x1, y1) = gen_arti(centerx=1, centery=1, sigma=0.05, nbex=100, data_type=0, epsilon=0.02)
(x2, y2) = gen_arti(centerx=1, centery=1, sigma=0.05, nbex=100, data_type=0, epsilon=0.02)
(log_w, log_f) = p.fit(x1,y1)
print p.predict
plot_frontiere(x2, p.predict,step=20)
plot_data(x2, y2)


#==============================================================================
#  Recuperartion des deux classes
#==============================================================================

def deux_classes(datax,datay,c1,c2):
    class1=[]
    class2=[]
    for i in range(len(datax)):
        if datay[i]==c1:
            class1.append(datax[i])
        if datay[i]==c2:
            class2.append(datax[i])
    classe1 = np.array(class1)
    classe2 = np.array(class2)
    
    x1 = np.concatenate((classe1,classe2))
    y1 = np.concatenate(([1]*len(classe1),[-1]*len(classe2)))
    
    return x1, y1

#==============================================================================
# Teste avec 2 classes 
#==============================================================================  
  

datax , datay = load_usps("USPS_train.txt")  
x1,y1 = deux_classes(datax,datay,c1=6,c2=9)          
p = Perceptron(loss=hinge,loss_g=hinge_grad, max_iter=100, eps=0.01)    
(log_w, log_f) = p.fit(x1,y1)
w = p.w
plt.figure(2)
plt.hist(w)# histogramme des poids w


########################

p = Perceptron(loss=hinge,loss_g=hinge_grad, max_iter=100, eps=0.01)
# Pour le cas stockastique
#p = Perceptron(loss=hinge,loss_g=hinge_grad_s, max_iter=100, eps=0.01)
(x1, y1) = gen_arti(centerx=1, centery=1, sigma=0.05, nbex=100, data_type=1, epsilon=0.02)
(x2, y2) = gen_arti(centerx=1, centery=1, sigma=0.05, nbex=100, data_type=1, epsilon=0.02)
(log_w, log_f) = p.fit(x1,y1)
plt.figure(3)
plot_frontiere(x2, p.predict,step=20)
plot_data(x2, y2)


##########################

p = Perceptron(loss=hinge,loss_g=hinge_grad, max_iter=100, eps=0.01, p = "prog3D")
# Pour le cas stockastique
#p = Perceptron(loss=hinge,loss_g=hinge_grad_s, max_iter=100, eps=0.01)
(x1, y1) = gen_arti(centerx=1, centery=1, sigma=0.05, nbex=100, data_type=1, epsilon=0.02)
(x2, y2) = gen_arti(centerx=1, centery=1, sigma=0.05, nbex=100, data_type=1, epsilon=0.02)
(log_w, log_f) = p.fit(x1,y1)
plt.figure(4)
plot_frontiere(x2, p.predict,step=20)
plot_data(x2, y2)



(log_w, log_f) = p.fit(x1,y1)
plot_frontiere(x2, p.predict,step=20)
print "test"
print log_w[-1]

# (x1 , x2) -> (x1,x2, x1*x2)
print y1
print x1

indexnoir = np.where(y1==-1)[0]
print indexnoir

indexbleu = np.where(y1==1)[0]
        
xnoir = np.array([x1[i] for i in indexnoir])  
print xnoir    
xbleu = np.array([x1[i] for i in indexbleu])  
plt.figure(5)    
ax = plt.gca(projection='3d')


#c=b2,s=1500,marker='*',edgecolors='none',depthshade=0
ax.scatter(xnoir[:,0],xnoir[:,1], xnoir[:,0]*xnoir[:,1],c="red",marker='*',edgecolors='none',depthshade=0)
ax.scatter(xbleu[:,0],xbleu[:,1], xbleu[:,0]*xbleu[:,1],c="blue",marker='*',edgecolors='none',depthshade=0)

grid,xx,yy = make_grid(xmin=-2 , xmax=2 , ymin = -2 , ymax=2)
print xx
print len(yy)

res= np.array( xx[:]*log_w[-1][1] + yy[:]*log_w[-1][2]+ log_w[-1][0])
print "test"
print res
surf = ax.plot_surface(xx,yy,res,rstride=1,cstride=1,\
	  cmap=cm.gist_rainbow, linewidth=0, antialiased=False)

