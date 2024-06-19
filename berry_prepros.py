import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import time

#-------- SIMULACIÓN ENTRADA DE OSCILOSCOPIO 
#CARGAR BASE DE DATOS  --  Se cargan las bases de datos de validación de la red neuronal para que no entren datos de entrenamiento
X_pruebas = np.load("X_pruebas.npy") #datos para validación
Y_pruebas = np.load("Y_pruebas.npy") #etiquetas correctas de validación

new_model = keras.models.load_model('modelo_conv2')
keras.Input(shape=[], dtype = tf.float32)

count = 0
while(True): #Por ahora recorre uno a uno y muestra la prediccion cada segundo
    y_res = new_model.predict(X_pruebas[count:count+1])
    y_res = y_res.round()
    print(y_res.argmax(axis=1) == Y_pruebas[count:count+1].argmax(axis=1))
    time.sleep(0.5)
    if count >= X_pruebas.shape[0]: #vuelve a recorrer la base de datos
        count = 0
    else:
        count += 1

            
