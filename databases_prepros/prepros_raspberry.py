import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle

def gest_to_matriz(lista):

    matrix = np.zeros((1,8))
    tam = []
    for gesto in lista:
        aux = np.array(gesto)
        tam.append(len(gesto))
        matrix = np.append(matrix, aux, axis=0)

    #Elimino la primera fila
    matrix = np.delete(matrix, 0, axis=0)
    return matrix


#leo la lista de gestos para post validación
with open("databases_prepros/datos_raspberry/X_berry.pickle", "rb") as f:
    x_post = pickle.load(f)

y_post = np.load('databases_prepros/datos_raspberry/Y_berry.npy')

#Compruebo correcta carga de datos de postvalidación
print(len(x_post))
print(y_post.shape[0])


dat = gest_to_matriz(x_post)


plt.figure()
plt.plot(dat[:,0])
plt.grid()
plt.show()

