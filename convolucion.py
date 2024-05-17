import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow import keras
import joblib
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

#imprime los tamaños de cada arreglo
print(label.shape)
print(dB.shape)


# #grafica el vector de etiquetas
# plt.figure()
# plt.grid()
# plt.plot(label)
# plt.show()

# #muestra 30 imagenes
# for fig in range(0,30):
#     plt.subplot(5,6,fig+1)
#     plt.title(label[fig])
#     plt.imshow(dB[fig],cmap=plt.cm.binary)

#plt.show()

#guarda las etiquetas
# np.savetxt("labels.csv", label, delimiter=",")

# #guarda las imagenes
# for i in range(864):
#     im = np.array(dB[i] * 255, dtype = np.uint8)
#     threshed = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
#     cv2.imwrite("img_datashet/"+str(i)+".png", threshed)


import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dB, label, test_size=0.2, shuffle=True)#separa y baraja los datos

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)) #convierte los archivos numpy a tf.dataset
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)) 



BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


#entrena la red y graba el historial de entrenamiento en la variable historial
historial = model.fit(train_dataset, epochs=500)



eval = model.evaluate(test_dataset)
print(eval)

joblib.dump(model, 'modelo_cnn5')

print("---------------RELOAD MODEL---------------")
#confirmar que si se reevalua da el mismo resultado
from joblib import load



clf = load('modelo_cnn5')

accuracy_reload_model = clf.evaluate(test_dataset)


print(accuracy_reload_model)

