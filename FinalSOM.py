# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 16:12:04 2022

@author: grego
Gregorio ALejandro Oropeza Gomez
"""

import numpy as np
from minisom import MiniSom
import imageio

classesprefix=['cars', 'roses']#el prefijo que identifica cada clase
data = []

print('Escriba la direccion del dataset, recuerde usar "/"')
Dir = input()
all_covers = glob(Dir +'/*.jpg')
    
for cover_jpg in all_covers:
    cover = imageio.imread(cover_jpg)
    data.append(cover.reshape(np.prod(cover.shape)))
    
original_shape = imageio.imread(all_covers[0]).shape

scaler = StandardScaler()
data = scaler.fit_transform(data)

data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

print('Elija la forma de la red')
print('Alto')
w = int(input())
print('Largo')
h = int(input())
som_shape = (h, w)

som = MiniSom(h, w, len(data[0]), sigma=0.75, learning_rate=.5,
              neighborhood_function='gaussian', random_seed=10)
print('Introdusca la cantidad de epocas')
epoch=input()
som.train_batch(data, epoch, verbose=True)
winner_coordinates = np.array([som.winner(x) for x in data]).T
cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)

with open('som.p', 'wb') as outfile:
    pickle.dump(som, outfile)

# ploting distances
for c in np.unique(cluster_index):
    plt.scatter(data[cluster_index == c, 0],
                data[cluster_index == c, 1], label='cluster='+str(c), alpha=.5)
    

# plotting centroids
for centroid in som.get_weights():
    plt.scatter(centroid[:, 0], centroid[:, 1], marker='x', 
                s=100, linewidths=0.5, color='k'  )#,label='centroid'
plt.legend();

#ploting frecuencies
plt.figure(figsize=(10, 10))
frequencies = som.activation_response(data)
plt.pcolor(frequencies.T, cmap='Reds') 
plt.colorbar()
plt.show()

#Determinining the class of each node
clusterSize=zeros([len(np.unique(cluster_index)), len(cluster_index)])

for i in range(len(np.unique(cluster_index))):#NO. nodes
    a=0
    for j in range(len(cluster_index)):#NO. inputs(images)
        if cluster_index[j]  == i:
            clusterSize[i,a]=j+1
            a=a+1

classesfreq=zeros([len(np.unique(cluster_index)),len(classesprefix)])#conunt of classes  frequency            

for i in range(len(np.unique(cluster_index))):#node num
    for j in range(len(cluster_index)):#inputs(images) num
        if clusterSize[i,j]!=0:
            imagedir=all_covers[int(clusterSize[i,j])-1]
            
            #image = Image.open(all_covers[clusterSize[i,j]-1])
            kaux=0
            for k in classesprefix:
                compareglob=glob(Dir +'/' +k +'*.jpg')
                for l in compareglob:
                    if l==imagedir:
                        classesfreq[i,kaux]=classesfreq[i,kaux]+1
                kaux=kaux+1

#recorrer la matriz de frecuencia y comparar cada celda para identificar a que neurona clase pertenece
for i in range(len(classesfreq)):
    winner_class=np.argmax(classesfreq[i,:])
    int(winner_class)
    print('Node ' +str(i) +' belongs to ' +str(classesprefix[winner_class]))