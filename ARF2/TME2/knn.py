# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 17:02:14 2017

@author: 3200183
"""

import  numpy  as  np
import  matplotlib.pyplot  as  plt
import  matplotlib.image  as  mpimg
import math as m


class method_knn():
    xmin ,xmax  =  2.23 ,2.48      ##  coord_x  min  et  max
    ymin ,ymax  =  48.806 ,48.916  ##  coord_y  min  et  max

        
    def predict(self,data,step,k):
        self.gapx = (self.xmax-self.xmin)/step
        self.gapy = (self.ymax- self.ymin)/step
        res = np.zeros((step,step))
        grad = self.list_grad(step)
        x=0
        for i in grad:
           x+=1
           y= 0
           for j in i:
                y+=1
                dist =self.list_dist(j[0],j[1],data)
                res[i,j]= self.res_case(k,dist,data)
        return res
            
    def list_grad(self,step):
        hist =  np.zeros((step,step,2))
        for i in range(1,step):
           for j in range(1,step):
               print ((self.xmin+self.gapx/2)+self.gapx*i,(self.ymin+self.gapy/2)+self.gapx*j)
               hist[i-1][j-1][0] = (self.xmin+self.gapx/2)+self.gapx*i
               hist[i-1][j-1][1] = (self.ymin+self.gapy/2)+self.gapy*i
        return hist
   
    def list_dist(self,x,y,data):
         res = np.zeros(len(data))
         cpt = 0
         for d in data:
             res[cpt] = m.sqrt(m.pow(x-d[0],2)+m.pow(y-d[1],2))
             cpt=cpt+1
         return res
      
    def res_case(self,k, dists,data):
          test = dists
          print type(test)
          res =0
          for i in range(k):
              ind =test.argmin(test) 
              res += data[ind][3]
              test.delete(ind)
          res =  res/k          
          if (res < 0.5 ):
              return 0
          else :
              return 1
              
            