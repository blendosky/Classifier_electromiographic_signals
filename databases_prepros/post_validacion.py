import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import ConfusionMatrixDisplay

#Carga las bases de datos de testeo o validacion 
X_pruebas = np.load("databases_prepros/dB_post.npy")
Y_pruebas = np.load("databases_prepros/y_test.npy")

#87 gestos son los de post validación y 777 de entrenamiento y validación
print(X_pruebas.shape)
print(Y_pruebas.shape)

X_pruebas = X_pruebas.reshape(X_pruebas.shape[0], 28, 28, 1)
X_pruebas = X_pruebas.astype('float32') 

Y_pruebas = to_categorical(Y_pruebas)

print('--------RECARGAR MODELO------')

new_model = keras.models.load_model('databases_prepros/modelo_conv5')

#valida el modelo con la base de datos de testeo
eval = new_model.evaluate(X_pruebas, Y_pruebas)


print(eval)


y_res = new_model.predict(X_pruebas)
y_res = y_res.round()



disp = ConfusionMatrixDisplay.from_predictions(Y_pruebas.argmax(axis=1), y_res.argmax(axis=1))
plt.show()