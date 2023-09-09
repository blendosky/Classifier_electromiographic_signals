import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
import cv2


datos = []
with open('datashets/1.txt') as fname:
	lineas = fname.readlines()
	for linea in lineas:
		datos.append(linea.strip('\n').split())

datos.pop(0)

dat = np.array(datos, dtype=float)
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
dat = scaler.fit_transform(dat)
res = cv2.resize(dat, dsize=(28, 28), interpolation=cv2.INTER_LINEAR)

print(res)
plt.figure()
plt.imshow(res,vmin=0,vmax=1)
plt.show()

#guardar cada txt como imagen y probar entrenando

