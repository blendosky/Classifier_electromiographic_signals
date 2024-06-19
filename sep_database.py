import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import pickle

#function of normalizated min max
def minmax_norm(df_input):
    return (df_input - df_input.min()) / ( df_input.max() - df_input.min())*2 - 1

#separa los conjuntos de datos de entrenamiento y validación, y post-procesamiento
def postvalidacion_data():
    #lee el archivo de texto
    frames = []
    for i in range(1,73):
        df = pd.read_csv("datashets/"+str(i)+".txt", sep="\t")
        frames.append(df)

    data = pd.concat(frames)

    data.drop(['time'], axis=1, inplace=True)

    data.reset_index(inplace=True, drop=True)

    #borra la clase 0 y 7
    data.loc[data['class'] == 7 , ['class']] = 0
    data.loc[data['class'] == 0 , ['channel1','channel2','channel3','channel4','channel5','channel6','channel7','channel8']] = 0
    data_class = data['class']
    data.drop(data_class.index[data_class == 0], axis=0, inplace=True)      
    data.reset_index(inplace=True, drop=True)

    #ahora si se deben separar por gestos para mantener la forma de la señal
    dataframe = data.to_numpy()
    filas = dataframe.shape[0]
    label = []
    clase = 0
    list_list = []
    for i in range(0,filas):
        if clase != dataframe[i,8]:
            list = []
            clase = dataframe[i,8]
            label.append(clase)
            list_list.append(list) 
        if clase == dataframe[i,8]:
            list.append(dataframe[i,:8])

    #Separación de conjuntos de datos y permutación aleatoria
    X_train, X_test, y_train, y_test = train_test_split(list_list, label, test_size=0.1, shuffle=True)


    return X_train, X_test, y_train, y_test

#Preprocesamiento de la base de datos
def preprocesamiento_rms(datos, etiquetas, div):

    gestos_reducidos = []

    #Aplica el RMS a cada gesto de forma independiente
    for gest,etiq in zip(datos,etiquetas):
        acum = np.zeros((div,8))
        cont = 0
        gest_reduc = np.zeros((1,9))
        gest_reduc = np.resize(gest_reduc,(1,8))
        for fil in range(0,len(gest)):
            if cont >= div:
                cont = 0
                aux = np.power(np.multiply(np.sum(np.power(acum,2), axis=0),1/div),0.5)
                gest_reduc = np.append(gest_reduc, [aux], axis=0)
            acum[cont,:] = gest[fil]
            cont += 1  
        gest_reduc = np.delete(gest_reduc, 0, axis=0)
        gestos_reducidos.append(gest_reduc)


    return gestos_reducidos

#separa el dataframe en gestos 
def sep_gestos(dataframe):
    filas = dataframe.shape[0]
    label = []
    clase = 0
    list_list = []
    for i in range(0,filas):
        if clase != dataframe[i,8]:
            list = []
            clase = dataframe[i,8]
            label.append(clase)
            list_list.append(list) 
        if clase == dataframe[i,8]:
            list.append(dataframe[i,:8])

    return list_list


def trans_tamaño(list_list, label):
    etiquetas = []
    dB = []
    for i in range(0,len(list_list)):
        aux = np.zeros((1,8))
        aux = np.delete(aux, 0, axis=0)
        for l in list_list[i]:
            aux = np.insert(aux, aux.shape[0] , l, 0)  
        if aux.shape[0] > 0:#Ajusta el tamaño de las matrices si tienen datos
            aux = cv2.resize(aux, dsize=(28, 28), interpolation=cv2.INTER_LINEAR)           
            dB.append(aux) 
            etiquetas.append(label[i]) #crea un nuevo vector de etiquetas con los gestos que se transformaron

    #Convierte en numpy's array
    dB = np.array(dB)
    etiquetas = np.array(etiquetas)

    return dB, etiquetas

X_train, X_test, y_train, y_test = postvalidacion_data()
y_train = np.array(y_train)
y_test = np.array(y_test)

#guardo los datos de postvalidacion sin preprocesar para el dispositivo hardware
print(len(X_test[0]))
with open("databases_prepros/datos_raspberry/X_berry.pickle", "wb") as f:
    pickle.dump(X_test, f)
np.save('databases_prepros/datos_raspberry/Y_berry.npy', y_test)

gest_ent = preprocesamiento_rms(X_train, y_train, 50)
gest_pos = preprocesamiento_rms(X_test, y_test, 50)


dB_train, y_train = trans_tamaño(gest_ent, y_train)
dB_post, y_test = trans_tamaño(gest_pos, y_test)


print(y_train.shape)
print(dB_train.shape)
print(y_test.shape)
print(dB_post.shape)

#guardar las bases de datos en formato numpy

#datos de entrenamiento
np.save("databases_prepros/dB_train.npy", dB_train)
np.save("databases_prepros/y_train.npy",y_train)
#datos de postvalidación
np.save("databases_prepros/dB_post.npy", dB_post)
np.save("databases_prepros/y_test.npy",y_test)





