import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow import keras

size = (28,28)

dataframe = pd.read_csv('database_reduc_rms.csv')
dataframe = dataframe.to_numpy()
filas, columnas = dataframe.shape
label = []
clase = 0
list_list = []
for i in range(filas):
    if clase == dataframe[i,8]:
        list.append(dataframe[i,:7])
    else:
        list = []
        clase = dataframe[i,8]
        label.append(clase)
        list_list.append(list)
    


dB = []
for i in range(0,len(list_list)):
    aux = np.zeros((1,7))
    aux = np.delete(aux, 0, axis=0)
    for l in list_list[i]:
        aux = np.insert(aux, aux.shape[0] , l, 0)  
    aux = cv2.resize(aux, dsize=(28, 28), interpolation=cv2.INTER_LINEAR)
    dB.append(aux)  

dB = np.array(dB)
# tamaño de dB (864,28,28)  hay 864 "imagenes" de 28x28

#array de etiquetas
label = np.array(label, dtype=int)

X_train, X_test, y_train, y_test = train_test_split(dB, label, test_size=0.2, shuffle=True)#separa y baraja los datos

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


X_entrenamiento = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_pruebas = X_test.reshape(X_test.shape[0], 28, 28, 1)

#hace que la salida sea un 1 logico en las neuronas de salida que tienen que coincidir con las etiquetas
Y_entrenamiento = to_categorical(y_train)
Y_pruebas = to_categorical(y_test)

#Convertir a flotante 
X_entrenamiento = X_entrenamiento.astype('float32') 
X_pruebas = X_pruebas.astype('float32') 

#Modulo de modificacion de "imagenes" en este caso las mueve horizontalmente 
rango_rotacion = 0
mov_ancho = 0.25
mov_alto = 0.0
rango_acercamiento=[1.0,1.0]

datagen = ImageDataGenerator(
    rotation_range = rango_rotacion,
    width_shift_range = mov_ancho,
    height_shift_range = mov_alto,
    zoom_range=rango_acercamiento,
    
)




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


modelo = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(7, activation="softmax")
])

#Compilación
modelo.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(X_entrenamiento.shape)
print(Y_entrenamiento.shape)

data_gen_entrenamiento = datagen.flow(X_entrenamiento, Y_entrenamiento, batch_size=32)


TAMANO_LOTE = 32


#Almacenar arreglos para cargar en el archivo de pruebas
np.save("X_pruebas.npy", X_pruebas)
np.save("Y_pruebas.npy", Y_pruebas)



print("Entrenando modelo...")
epocas=500
history = modelo.fit(
    data_gen_entrenamiento,
    epochs=epocas,
    batch_size=TAMANO_LOTE,
    validation_data=(X_pruebas, Y_pruebas)
)



print("Modelo entrenado!")

print(modelo.evaluate(X_pruebas, Y_pruebas))

plt.figure()
plt.plot(history.history["loss"])
plt.grid()
plt.show()

modelo.save('modelo_conv2')

print('--------RECARGAR MODELO------')

new_model = keras.models.load_model('modelo_conv2')


eval = new_model.evaluate(X_pruebas, Y_pruebas)


print(eval)