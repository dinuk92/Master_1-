# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ce script temporaire est sauvegardé ici :
/users/Etu3/3200183/.spyder2/.temp.py
"""

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
plt.close('all') 


fname = "dataVelib.pkl"
f= open(fname,'rb')
data = pkl.load(f)
f.close()
tab = np.array(0)



for station in data :
    arr = station['number']/1000
    if arr >= 1 and arr <= 20:
       # tab = np.append(tab,[station['alt'],station['position'],arr,station['bike_stands'],station['available_bike_stands']])
        tab = np.append(tab,station)
print tab
print type(tab)