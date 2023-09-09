import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from scipy import signal
from scipy.signal import butter


datos = []
with open('datashets/1.txt') as fname:
	lineas = fname.readlines()
	for linea in lineas:
		datos.append(linea.strip('\n'))

datos.pop(0)
datos.pop(0)

database=[]

for prof in datos:
	database.append(prof.split())
	
dat = []
for x in database:
	sublist = []
	for xin in x:
		sublist.append(float(xin))
	dat.append(sublist)



dat = np.array(dat, dtype=float)
#print(dat[:10,:2])#database complete ok, its name is dat
#dat=sklearn.utils.shuffle(dat)


print(dat)



X = []
y = []
#preprocesamiento
for fil in dat:
	if fil[9]!=0:
		X.append(fil[1:9])
		y.append(fil[9])

	

X = np.array(X, dtype=float)
y = np.array(y, dtype=float)

print(X)
print(y)

scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

X = scaler.fit_transform(X)
X = np.round(X,3)

frec_corte = 250 #low pass filter frecuency
sos = butter(15, frec_corte, btype="low", fs=1000, output="sos")

for i in range(1,8):
    filtd = np.array(signal.sosfilt(sos, X[:,i]))
    X[:,i] = filtd

def minmax_norm(df_input):
    return (df_input - df_input.min()) / ( df_input.max() - df_input.min()) * 2 - 1

X = minmax_norm(X)

print(X)

fig, axs = plt.subplots(2)
axs[0].grid()
axs[0].plot(X[:,7])#ejemplo señal canal 1
axs[1].plot(y)#ejemplo señal canal 1
plt.show()





model = Sequential()
model.add(Dense(150, input_dim=8, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(1, activation='linear'))

#Nadam fue el mejor optimizador que encontré
model.compile(loss='mean_squared_error',
 	      optimizer='Adam',
		  metrics = ["accuracy"])


model.fit(X,y, epochs=110)

score = model.evaluate(X,y)
print(score)
y_predict = model.predict(X).round()

plt.figure(2)
plt.grid()
plt.plot(y_predict,'r')#clasificación de la red neuronal
plt.plot(y,color = 'tab:green')#clasificación de la base de datos
plt.show()