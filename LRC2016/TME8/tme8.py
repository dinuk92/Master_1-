# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 01:33:27 2016

@author: dinuk
"""

import definitions as defi;

print defi.transpose['>']

def transposeList(L):
    trans = []
    for i in L :
        trans.append(defi.transpose[i])
    return trans
 
L  = ['m','o','s','ot']
print transposeList(L);  

def symetrieList(L):
    trans = []
    for i in L :
        trans.append(defi.symetrie[i])
    return trans

L  = ['m','o','s']
print symetrieList(L);  

def compose (r1,r2):
    if (r1 == '=' ):
        return [r2]
    if (r2 == '='):
        return [r1]
    
    if ((r1,r2) in defi.compositionBase.keys()  ):
        return list(defi.compositionBase[(r1,r2)])
    elif ( (defi.transpose[r2],defi.transpose[r1]) in defi.compositionBase.keys()):
        return transposeList(defi.compositionBase[(defi.transpose[r2],defi.transpose[r1])])
    elif ( (defi.symetrie[r1],defi.symetrie[r2]) in defi.compositionBase.keys()):
        return symetrieList(defi.compositionBase[(defi.symetrie[r1],defi.symetrie[r2])])
    else:
        return symetrieList(transposeList(defi.compositionBase[(defi.transpose[defi.symetrie[r2]],defi.transpose[defi.symetrie[r1]])]))

print compose ('m','d')
print compose ('ot','>')
print compose ('>','e')
print compose ('ot','m')

def compositionListe(L1, L2):
    res = []
    for r1 in L1:
        for r2 in L2 :
            for elt in compose(r1, r2):
                res.append(elt)
    return list(set(res)) 

print compositionListe(['m','o'],['=','dt','st','et'])

relation = {('l','s'):{'ot','mt'}, 
            ('s','r'):{'m','mt','<','>'},
            ('l','r'):{'<','>','o','m','d','s','st','e','='},
            }

def getrelation(r):
    n =set()
    for (r1,r2)in r :
        n.add(r1)
        n.add(r2)
    return list(n)

def propage(''):
    
    

print getrelation(relation)

#def Allen(m,r):