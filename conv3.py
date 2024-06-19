import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow import keras


#importamos los conjuntos de datos de entrenamiento
dB_train = np.load("databases_prepros/dB_train.npy")
label = np.load("databases_prepros/y_train.npy")


#Separamos los datos de entrenamiento en un 80% para entrenar y un 20% para validar --> El 80% de los gestos son 777
X_train, X_test, y_train, y_test = train_test_split(dB_train, label, test_size=0.1, shuffle=True)

print("Conjunto de entrenamiento:", X_train.shape)
print("Conjunto de validación:", X_test.shape)


#Ajustamos el tamaño de los conjuntos de entrenamiento y validación, de forma que queden de 28*28*1
X_entrenamiento = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_pruebas = X_test.reshape(X_test.shape[0], 28, 28, 1)
#Convertirmos los conjuntos de entrenamiento y validación a formato flotante de 32 bits
X_entrenamiento = X_entrenamiento.astype('float32') 
X_pruebas = X_pruebas.astype('float32') 

#Ajustamos el formato del conjunto de etiquetas de forma que quede en formato de matriz de unos y no de vector
Y_entrenamiento = to_categorical(y_train)
Y_pruebas = to_categorical(y_test)

#Caracteristicas del data aumentation
rango_rotacion = 0
mov_ancho = 0.1
mov_alto = 0 #0.05 97%
rango_acercamiento=[1.0,1.0]

#Cargar las caracteristicas en el data aumentation
datagen = ImageDataGenerator(
    rotation_range = rango_rotacion,
    width_shift_range = mov_ancho,
    height_shift_range = mov_alto,
    # zoom_range=rango_acercamiento,   
)

#Graficar los datos normales con etiquetas y con el data aumentation
filas = 4
columnas = 8
num = filas*columnas
print('ANTES:\n')
fig1, axes1 = plt.subplots(filas, columnas, figsize=(1.5*columnas,2*filas))
for i in range(num):
     ax = axes1[i//columnas, i%columnas]
     ax.imshow(X_entrenamiento[i].reshape(28,28), cmap='gray_r')
     ax.set_title('Label: {}'.format(np.argmax(Y_entrenamiento[i])))
plt.tight_layout()
plt.show()
print('DESPUES:\n')
fig2, axes2 = plt.subplots(filas, columnas, figsize=(1.5*columnas,2*filas))
for X, Y in datagen.flow(X_entrenamiento,Y_entrenamiento.reshape(Y_entrenamiento.shape[0], 7),batch_size=num,shuffle=False):
     for i in range(0, num):
          ax = axes2[i//columnas, i%columnas]
          ax.imshow(X[i].reshape(28,28), cmap='gray_r')
          ax.set_title('Label: {}'.format(int(np.argmax(Y[i]))))
     break
plt.tight_layout()
plt.show()


#Caracteristicas del modelo
modelo = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)), #32
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), #64
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.5), #Dropout 0.65 mejor encontrado
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'), #100
    tf.keras.layers.Dense(7, activation="softmax")
])

#Compilar el modelo
modelo.compile(optimizer='adam',
              loss='categorical_focal_crossentropy',
              metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ])

print(X_entrenamiento.shape)
print(Y_entrenamiento.shape)

#crea el conjunto de datos de entrenamiento a partir del datagen
data_gen_entrenamiento = datagen.flow(X_entrenamiento, Y_entrenamiento, batch_size=32)

TAMANO_LOTE = 32

#Entrenamos el modelo
print("Entrenando modelo...")
epocas=1000
history = modelo.fit(
    data_gen_entrenamiento,
    epochs=epocas,
    batch_size=TAMANO_LOTE,
    validation_data=(X_pruebas, Y_pruebas)
)

#Imprime la precision del modelo y grafica la curva de error por epocas
print("Modelo entrenado!")
print(modelo.evaluate(X_pruebas, Y_pruebas))
plt.figure()
plt.plot(history.history["loss"])
plt.grid()
plt.show()

#Guarda el modelo
modelo.save('databases_prepros/modelo_conv5')
