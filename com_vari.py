import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



dataframe = pd.read_csv('database2_reduc.csv') #lee el dataframe

dataframe = dataframe.to_numpy()
filas, columnas = dataframe.shape
label = []
clase = 0
list_list = []
for i in range(filas):
    if clase == dataframe[i,8]:
        list.append(dataframe[i,:8])
    else:
        list = []
        clase = dataframe[i,8]
        label.append(clase)
        list_list.append(list)


muestra_ges = []

for gest in range(1,7):
    aux_gest = []
    for i in range(len(list_list)): #Recorre el array de gestos
        if label[i] == gest:
            aux_gest.append(list_list[i])
    muestra_ges.append(aux_gest)

var_prom = np.zeros((1,8))

matriz_var = []


for gest in muestra_ges:
    col_var = []
    for i in range(len(gest)):
        if len(gest[i]) > 0:
            var_ges = np.array(gest[i]).var(axis=0)
            prom_var = sum(var_ges)/len(var_ges)
            col_var.append(prom_var)
    matriz_var.append(col_var)


print(np.array(matriz_var[1]))


matriz_var1 = np.load('varianza_db1.npy')

fig = plt.figure()
gs = fig.add_gridspec(2, hspace=0.4)
ax = gs.subplots(sharex=True)
ax[0].set_title("Varianza promedio por gesto DB1")
ax[0].set_xlabel("Gesto")
ax[0].set_ylabel("Varianza")
ax[0].grid()
ax[0].boxplot(x = matriz_var1)
ax[1].set_title("Varianza promedio por gesto DB2")
ax[1].set_xlabel("Gesto")
ax[1].set_ylabel("Varianza")
ax[1].grid()
ax[1].boxplot(x=matriz_var)
plt.show()
