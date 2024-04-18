import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import numpy as np
from sklearn.metrics import confusion_matrix

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow_datasets as tfds

dataframe = pd.read_csv('database_reduc.csv')

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

#re normalizacion total de todo el dataframe 0-1

scaled_df = scaler.fit_transform(dataframe)
scaled_df = pd.DataFrame(scaled_df, columns=['channel1', 'channel2', 'channel3','channel4', 'channel5', 'channel6', 'channel7', 'channel8', 'class'])


X = scaled_df[['channel1','channel2','channel3','channel4','channel5','channel6','channel7','channel8']].to_numpy(dtype=float)
y = dataframe['class'].to_numpy(dtype=float)




plt.figure(1)
plt.plot(X[:,0])
plt.plot(y)
plt.grid()
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
y_train = y_train.astype(int)
y_test = y_test.astype(int)

X_train = X_train
y_train = y_train

#sklearn clasifier
model = MLPClassifier(hidden_layer_sizes=(75,50,25), max_iter=1200, alpha=1e-7,
                    solver='adam', verbose=True, random_state=1,
                    learning_rate_init=.01, tol=1e-5, n_iter_no_change=100, activation='relu', shuffle = True)

historial = model.fit(X_train, y_train)


y_predict = model.predict(X_test)

color = np.where(y_predict != y_test, "green", "red")


fig, ax = plt.subplots()
ax.scatter(y_predict, y_test, c = color, alpha=0.01)
plt.grid()
plt.show()


precision = accuracy_score(y_test, y_predict)

print(precision)


plt.figure(2)
plt.plot(historial.loss_)
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.show




