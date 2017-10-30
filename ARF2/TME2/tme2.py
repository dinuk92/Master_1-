# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 10:45:05 2017

@author: 3200183
"""

import  numpy  as  np
import  matplotlib.pyplot  as  plt
import  matplotlib.image  as  mpimg
import method_histo 
import method_parzen
import knn
fname  =  "velib.npz"
##  fonction  de  d e s e r i a l i z a t i o n  de  numpy
obj  =  np.load( fname )
##  objets  contenus  dans  l e   f i c h i e r
print ( obj.keys ( ) )
##  matrice  1217x#minutes  nombre  de  velos  disponibles
histo  =  obj['histo']
##  matrice  1217x#minutes ,  pour  chaque  station  nombre  de  v e l i b   pris  a  chaque  minute
take  =  obj['take']
##  infos   stations  statiques :
####  id_velib ->(nom, addresse , coord_y , coord_x , banking , bonus , nombre  de  places
stations  =  dict(obj['stations'].tolist())
##  i d _ v e l i b->  id  matrice  take,histo
idx_stations  =  dict( obj ['idx_stations'].tolist())
stations_idx  =  dict( obj ['stations_idx'].tolist())   ##  id  matrice->  v e l i b

data = []
for i in idx_stations.values():
    data.append([stations[i][2],stations[i][3],stations[i][4]])
print data
#pirnt data = [stations[idx_stations[]][2],stations[idx_stations[1]][3],stations[idx_stations[1]][6]
    
#print histo

plt.ion()
parismap  =  mpimg. imread ( 'paris.jpg' )
##  coordonnees  GPS  de  la   carte
xmin ,xmax  =  2.23 ,2.48      ##  coord_x  min  et  max
ymin ,ymax  =  48.806,48.916  ##  coord_y  min  et  max

def  show_map ( ) :
    plt.imshow ( parismap , extent =[xmin ,xmax, ymin ,ymax] , aspect =1.5)
    ##  extent  pour  controler   l ’ echelle  du  plan
    
geo_mat  =  np . zeros ( ( len ( idx_stations ) , 2 ) )
for   i , idx  in  idx_stations.items() :
    geo_mat [i,0]=stations[idx][3]
    geo_mat [i,1]=stations[idx][2]
##  alpha  permet  de  r e g l e r   la  transparence
plt.scatter(geo_mat[:,0],geo_mat[:,1],alpha =0.3)
    
#show_map()

steps=100
xx , yy  =  np.meshgrid( np.linspace ( xmin ,xmax,steps ) , np.linspace( xmin ,xmax, steps ) )
#print xx 
#print yy
grid  =  np.c_[ xx.ravel(), yy.ravel()]
#print grid

#res =method_histo.method_histo()
#res= method_parzen.method_parzen()
res= knn.method_knn()
r = res.predict(data,steps,1).reshape(steps,steps)
#r = res.predict(data,steps).reshape(steps,steps)
#print r
#res=np.random.random((steps,steps))
show_map()
plt.imshow(r,extent=[xmin ,xmax, ymin ,ymax] , interpolation = 'none' , alpha =0.3 , origin  =  " lower " )
plt.colorbar()
#plt.scatter(geo_mat[:,0],geo_mat[:,1],alpha =0.3)
