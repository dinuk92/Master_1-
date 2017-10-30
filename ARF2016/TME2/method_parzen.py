# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 13:32:45 2017

@author: 3200183
"""

import  numpy  as  np
import  matplotlib.pyplot  as  plt
import  matplotlib.image  as  mpimg

class method_parzen():
    xmin ,xmax  =  2.23 ,2.48      ##  coord_x  min  et  max
    ymin ,ymax  =  48.806 ,48.916  ##  coord_y  min  et  max

        
    def predict(self,data,step):
        self.gapx = (self.xmax-self.xmin)/step
        self.gapy = (self.ymax- self.ymin)/step
        hist =  np.zeros((step,step))
        cpt=0
        for d in data :
            print d
            i,j = self.to_bin(d[1],d[0])
            print "test ij "
            print i 
            print j
            hist[i,j] +=1
            #cpt = cpt + 1
        hist = hist/step
        return hist
        
    def frange(self,x, y, jump):
        while x < y:
            yield x
            x += jump

        
    def to_bin(self,x, y) :
        i =0
        j= 0
        depx=self.xmin+self.gapx/2
        depy=self.ymin+self.gapy/2
        for k in list(self.frange( depx, self.xmax , self.gapx)):
            i = i+1
            print "test"
            print np.abs(k-x)
            if (np.abs(k-x)<=0.5 ):
                   for l in list(self.frange(depy  , self.ymax , self.gapy)):
                       j = j+1
                       if (np.abs(l-y)<=0.5 ):
                           return i,j