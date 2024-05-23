import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import numpy as np
import tensorflow as tf


#function of normalizated min max
def minmax_norm(df_input):
    return (df_input - df_input.min()) / ( df_input.max() - df_input.min())*2 - 1
# Create a dataframe with all the data without preprocessing


data = pd.read_csv('BD2.csv')


data.reset_index(inplace=True, drop=True)

print(data.shape)

data.loc[data['class'] == 7 , ['class']] = 0



data[['channel1','channel2','channel3','channel4','channel5','channel6','channel7','channel8']] = minmax_norm(data[['channel1','channel2','channel3','channel4','channel5','channel6','channel7','channel8']])
data.loc[data['class'] == 0 , ['channel1','channel2','channel3','channel4','channel5','channel6','channel7','channel8']] = 0

data_class = data['class']

data.drop(data_class.index[data_class == 0], axis=0, inplace=True)
       
data.reset_index(inplace=True, drop=True)

print(data.shape)

#agregar el rms y prueba de entrenamiento
filas, columnas = data.shape
div = 50

iter = 0
data_comp = pd.DataFrame(columns = ['channel1','channel2','channel3','channel4','channel5','channel6','channel7','channel8', 'class'])
print(data_comp)
for i in range(1,int(filas/div)+1):
    data_bloque = data.iloc[iter:(i*div),:].pow(2)
    data_bloque = ((data_bloque.sum()).div(div)).pow(0.5)
    iter = iter + div
    data_comp = pd.concat([data_comp, pd.DataFrame([data_bloque])], ignore_index=True)

#plotea el canal con rms, comparandolo con la clase --> el nivel de reduccion depende de la variable div
plt.figure()
plt.plot(data_comp['channel1'])
plt.plot(data_comp['class'])
plt.grid()
plt.show()

print(data_comp.shape)

#quitar el vrms directamente de los saltos (no da numeros enteros, agregando otras clases incorrectas)
data_comp['compare'] = data_comp['class'].apply(lambda x: True if x == 1 or x == 2 or x == 3 
                                                or x == 4 or x == 5 or x == 6 else False)
compare = data_comp['compare']
data_comp.drop(compare.index[compare == False], axis=0, inplace=True)  
data_comp.drop(['compare'], axis=1, inplace=True)
data_comp.reset_index(inplace=True, drop=True)




#guardar el dataframe
data_comp.to_csv('database2_reduc.csv', header=True, index=False)
print(data_comp.shape)