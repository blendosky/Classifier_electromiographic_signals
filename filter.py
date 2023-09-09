from scipy.signal import butter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from keras.models import Sequential
from keras.layers import Dense


data = pd.read_csv("dataframe_nopreprocesado.csv")
data = data.drop(["Unnamed: 0"], axis=1) #borra una fila que se crea cuando se lee el csv

#data = data.iloc[100:140000]

#balancear base de datos....

fig, ax = plt.subplots(3,1)
ax[0].plot(data['class'])
ax[1].plot(data['channel1'])
ax[0].grid()
ax[1].grid()

frec_corte = 350 #low pass filter frecuency
sos = butter(15, frec_corte, btype="low", fs=1000, output="sos")

for i in range(1,9):
    filtd = pd.DataFrame(signal.sosfilt(sos, data["channel"+str(i)]))
    data["channel"+str(i)] = filtd



#data ya tiene los canales filtrados, pasabajos con frecuencia de corte seleccionada frec_corte

ax[2].plot(data["channel1"],"r")
ax[2].set_title("After applying "+str(frec_corte)+" Hz high-pass filter")
ax[2].set_xlabel('pos_array')
ax[2].grid()
plt.tight_layout()
plt.show()

#Entrenamiento red neuronal

X = pd.DataFrame(round(data[['channel1','channel2','channel3','channel4','channel5','channel6','channel7','channel8']],3))
y = pd.DataFrame(data['class'])

a = y.isnull()
print(a)
rta = a[a == True]

#normalizacion function
def minmax_norm(df_input):
    return (df_input - df_input.min()) / ( df_input.max() - df_input.min()) *2 - 1

#normalizacion
X = minmax_norm(X)
print(np.shape(X))

#mejoro el que se generara loss nan
print(X)
print("--------------")
print(y)
print("-------Entrenando------------")


fig2, ax1 = plt.subplots(2,1)
ax1[0].plot(X['channel1'])
ax1[1].plot(y)
ax1[0].grid()
ax1[1].grid()
plt.show()



model = Sequential()
model.add(Dense(150, input_dim=8, activation='tanh'))
model.add(Dense(75, activation='tanh'))
model.add(Dense(1, activation='linear'))

#Nadam fue el mejor optimizador que encontré
model.compile(loss='mean_squared_error',
 	      optimizer='Adam',
          metrics=["accuracy"])


model.fit(X,y, epochs=10)

score = model.evaluate(X,y)
print(score)
y_predict = model.predict(X).round()

print("--------------Fin de entrenamiento------------------------")


plt.figure(2)
plt.grid()
plt.plot(y_predict,'r')#clasificación de la red neuronal
plt.plot(y,color = 'tab:green')#clasificación de la base de datos
plt.show()