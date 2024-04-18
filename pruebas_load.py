import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras


#Carga las bases de datos de testeo o validacion 
X_pruebas = np.load("X_pruebas.npy")
Y_pruebas = np.load("Y_pruebas.npy")

print(X_pruebas.shape)


print('--------RECARGAR MODELO------')

new_model = keras.models.load_model('modelo_conv2')

#valida el modelo con la base de datos de testeo
eval = new_model.evaluate(X_pruebas, Y_pruebas)


print(eval)


y_res = new_model.predict(X_pruebas)
y_res = y_res.round()


print(y_res==Y_pruebas)