import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense

datos = []
with open('datashets/1.txt') as fname:
	lineas = fname.readlines()
	for linea in lineas:
		datos.append(linea.strip('\n'))

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



X = []
y = []
#preprocesamiento
for fil in dat:
	if fil[9]!=0:
		X.append(fil[1:8])
	elif fil[9]==0:
		X.append(fil[1:8]*0)

X = np.array(X, dtype=float)


y = dat[:,9].reshape(-1,1)

scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

X = scaler.fit_transform(X)
X = np.round(X,3)

fig, axs = plt.subplots(2)
axs[0].grid()
axs[0].plot(dat[:,0],X[:,2])#ejemplo señal canal 1
axs[1].plot(dat[:,0],y)#ejemplo señal canal 1
plt.show()



model = Sequential()
model.add(Dense(80, input_dim=7, activation='tanh'))
model.add(Dense(40, activation='tanh'))
model.add(Dense(25, activation='relu'))

#Nadam fue el mejor optimizador que encontré
model.compile(loss='mean_squared_error',
 	      optimizer='Nadam',
		  metrics = ["accuracy"])


model.fit(X,y, epochs=500)

score = model.evaluate(X,y)
print(score)
y_predict = model.predict(X).round()

plt.figure(2)
plt.grid()
plt.plot(dat[:,0],y_predict)#clasificación de la red neuronal
plt.plot(dat[:,0],y,color = 'tab:green')#clasificación de la base de datos
plt.show()