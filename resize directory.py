# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 15:23:11 2022

@author: grego
"""


import os
from PIL import Image
print('Introdisca la direccion donde se encuantran las imagenes a normalizar')
maindir = input()
maindir=maindir +'/'
contenido = os.listdir(maindir)
imagenes = []
for fichero in contenido:
    if os.path.isfile(os.path.join(maindir, fichero)) and fichero.endswith('.jpg'):
        imagenes.append(fichero)
        image = Image.open(maindir +fichero)
        new_image = image.resize((250, 250))
        new_image.save('r-' +fichero)

