from sklearn.model_selection import train_test_split
import pandas as pd
import csv
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds





X = np.random.rand(100,3)
y = np.random.randint(0,2, (100))

# Create tf.data.Dataset from random data
train_dataset = tf.data.Dataset.from_tensor_slices((X,y))
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)


datos, metadatos = tfds.load('mnist', as_supervised=True, with_info=True)

datos_entrenamiento = datos['train']


plt.figure()
for img in train_dataset:
    
    plt.imshow(img, cmap=plt.cm.binary)


plt.show()