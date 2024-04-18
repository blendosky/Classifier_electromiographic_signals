from joblib import load
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

clf = load('regression.joblib')

data = pd.read_csv("dataframe_nopreprocesado.csv")
data = data.drop(["Unnamed: 0"], axis=1) #borra una fila que se crea cuando se lee el csv

#data = data.iloc[100:54000]
#data = data.iloc[497000:1000000]



fig, ax = plt.subplots(2,1)
ax[0].plot(data['class'])
ax[1].plot(data['channel1'])
ax[0].grid()
ax[1].grid()
plt.show()



X = data[['channel1','channel2','channel3','channel4','channel5','channel6','channel7','channel8']] 
y = data['class']
X.reset_index(inplace=True, drop=True)
y.reset_index(inplace=True, drop=True)
y.fillna(1, inplace=True)
a = y.isnull()
print(a)
rta = a[a == True]
 
print(rta)

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