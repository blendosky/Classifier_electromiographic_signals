import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from joblib import dump
from matplotlib.colors import ListedColormap
from scipy.signal import butter
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from scipy import signal
from sklearn.metrics import precision_score

data = pd.read_csv("dataframe_nopreprocesado.csv")
data = data.drop(["Unnamed: 0"], axis=1) #borra una fila que se crea cuando se lee el csv

data = data.iloc[0:250000]

data.reset_index(inplace=True, drop=True)



# frec_corte = 250 #low pass filter frecuency
# sos = butter(15, frec_corte, btype="low", fs=1000, output="sos")

# for i in range(1,9):
#     filtd = pd.DataFrame(signal.sosfilt(sos, data["channel"+str(i)]))
#     data["channel"+str(i)] = filtd

fig, ax = plt.subplots(2,1)
ax[0].plot(data['class'])
ax[1].plot(data['channel1'])
ax[0].grid()
ax[1].grid()
plt.show()


X = data[['channel1','channel2','channel3','channel4','channel5','channel6','channel7','channel8']] 
y = data['class']



y.fillna(1, inplace=True)
#a = y.isnull()
#print(a)
#rta = a[a == True]

clf = MLPClassifier(solver='adam', alpha=1e-5,
                    hidden_layer_sizes=(200,150,20,5),
                    random_state=1, shuffle=True, verbose=True, 
                    learning_rate='invscaling', max_iter=250)



print('Entrenando...')
clf.fit(X,y)
print('Comprobando')
y_classifier = clf.predict(X)

accuracy = accuracy_score(y, y_classifier)
precision = precision_score(y, y_classifier, average='micro')

print("accuracy " + str(accuracy))
print("precisicion " + str(precision))

plt.figure(2)
plt.grid()
plt.plot(y,'g',linewidth=6)
plt.plot(y_classifier,'r')
plt.show()
print("guardando modelo...")
dump(clf, 'regression.joblib') 


