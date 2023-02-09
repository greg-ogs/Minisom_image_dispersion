# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 22:15:27 2022

@author: grego
Gregorio ALejandro Oropeza Gomez
"""

from minisom import MiniSom
import numpy as np
from numpy import zeros
import pandas as pd
import matplotlib.pyplot as plt
import sys
import imageio
from glob import glob
from sklearn.preprocessing import StandardScaler
from PIL import Image
import os
import pickle
#seleccion de classes
print('Escriba el numero de clases')
classescount=int(input())
classesprefix=[None for _ in range(classescount)]#el prefijo que identifica cada clase


#for i in range(classescount):
#    print('Escriba el nombre de la clase ' +str(i+1))
#    classescount[i]=input()

data = []
print('Introdusca la direccion de la(s) imagenes a identificar, recuerde usar "/" ')
Dir = input()
all_covers = glob(Dir +'/*.jpg')
    
for cover_jpg in all_covers:
    cover = imageio.imread(cover_jpg)
    data.append(cover.reshape(np.prod(cover.shape)))
    
original_shape = imageio.imread(all_covers[0]).shape

scaler = StandardScaler()
data = scaler.fit_transform(data)

print('Introdusca la direccion del modelo a usar, recuerde usar "/"')
modelDir = input()

with open(modelDir, 'rb') as infile:
    som = pickle.load(infile)

#select shape (cantidad de nodos)
print('Elija la forma de la red')
print('Alto')
w = int(input())
print('Largo')
h = int(input())
som_shape = (h, w)

som.train_batch(data, 100, verbose=True)
winner_coordinates = np.array([som.winner(x) for x in data]).T
cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)
#check cluster_index variable to know the winner node
for i in range(len(cluster_index)):
    print('Picture ' +str(all_covers[i])  +' belongs to node ' +str(cluster_index[i]))
    

