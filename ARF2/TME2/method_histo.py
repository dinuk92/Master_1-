# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 11:30:52 2017

@author: 3200183
"""

import  numpy  as  np
import  matplotlib.pyplot  as  plt
import  matplotlib.image  as  mpimg


class method_histo():
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
            hist[i,j] +=d[2] 
            cpt = cpt + d[2]
        hist = hist/cpt
        return hist
            
    def to_bin(self,x, y) :
        print self.gapx
        print self.gapy
        print "test gapx gapy"
        resx = (x-self.xmin)/self.gapx
        resy = (y-self.ymin)/self.gapy
        return resx , resy
            