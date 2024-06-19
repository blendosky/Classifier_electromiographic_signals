import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



dataframe = pd.read_csv('database_reduc.csv') #lee el dataframe

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
            for fil in list_list[i]:
                aux_gest.append(fil)
    muestra_ges.append(aux_gest)

vec_prom = np.zeros((1,8))


for i in range(len(muestra_ges)):
    val = (np.array(muestra_ges[i]))
    prom = []
    for col in range(val.shape[1]):
        prom.append(sum(val[:,col])/val.shape[0]) #promedio de la amplitud de cada canal de cada gesto
    vec_prom = np.insert(vec_prom, vec_prom.shape[0], prom, 0)   
        



width = 0.06

vec_prom = vec_prom[1:,:]

print(vec_prom.shape)
print(vec_prom)


np.save('ampl_prom_db1.npy', vec_prom)

db2 = np.load('ampl_prom_db2.npy')



fig = plt.figure()
gs = fig.add_gridspec(2, hspace=0.4)
ax = gs.subplots(sharex=True)
ax[0].set_title('Amplitud promedio de cada canal por gesto de la base de datos 1')
ax[0].set_ylabel('Amplitud promedio')
ax[0].set_xlabel('Gestos')
ax[0].grid()
ax[0].bar(np.arange(1,7,1)-0.4,vec_prom[:,0], width = width)
ax[0].bar(np.arange(1,7,1)-0.3,vec_prom[:,1], width = width)
ax[0].bar(np.arange(1,7,1)-0.2,vec_prom[:,2], width = width)
ax[0].bar(np.arange(1,7,1)-0.1,vec_prom[:,3], width = width)
ax[0].bar(np.arange(1,7,1),vec_prom[:,4], width = width)
ax[0].bar(np.arange(1,7,1)+0.1,vec_prom[:,5], width = width)
ax[0].bar(np.arange(1,7,1)+0.2,vec_prom[:,6], width = width)
ax[0].bar(np.arange(1,7,1)+0.3,vec_prom[:,7], width = width)
ax[0].legend(['Canal 1', 'Canal 2', 'Canal 3', 'Canal 4', 'Canal 5', 'Canal 6', 'Canal 7', 'Canal 8'], loc = 'best', fontsize="6")
ax[1].set_title('Amplitud promedio de cada canal por gesto de la base de datos 2')
ax[1].set_ylabel('Amplitud promedio')
ax[1].set_xlabel('Gestos')
ax[1].grid()
ax[1].bar(np.arange(1,7,1)-0.4,db2[:,0], width = width)
ax[1].bar(np.arange(1,7,1)-0.3,db2[:,1], width = width)
ax[1].bar(np.arange(1,7,1)-0.2,db2[:,2], width = width)
ax[1].bar(np.arange(1,7,1)-0.1,db2[:,3], width = width)
ax[1].bar(np.arange(1,7,1),db2[:,4], width = width)
ax[1].bar(np.arange(1,7,1)+0.1,db2[:,5], width = width)
ax[1].bar(np.arange(1,7,1)+0.2,db2[:,6], width = width)
ax[1].bar(np.arange(1,7,1)+0.3,db2[:,7], width = width)
ax[1].legend(['Canal 1', 'Canal 2', 'Canal 3', 'Canal 4', 'Canal 5', 'Canal 6', 'Canal 7', 'Canal 8'], loc = 'best', fontsize="6")
plt.show()


diff = abs(vec_prom - db2)

plt.figure()
plt.grid()
plt.title('Diferencia de amplitud promedio de canales por gesto entre bases de datos')
plt.xlabel('Gestos')
plt.ylabel('Diferencia de amplitud')
plt.bar(np.arange(1,7,1)-0.4,diff[:,0], width = width)
plt.bar(np.arange(1,7,1)-0.3,diff[:,1], width = width)
plt.bar(np.arange(1,7,1)-0.2,diff[:,2], width = width)
plt.bar(np.arange(1,7,1)-0.1,diff[:,3], width = width)
plt.bar(np.arange(1,7,1),diff[:,4], width = width)
plt.bar(np.arange(1,7,1)+0.1,diff[:,5], width = width)
plt.bar(np.arange(1,7,1)+0.2,diff[:,6], width = width)
plt.bar(np.arange(1,7,1)+0.3,diff[:,7], width = width)
plt.legend(['Canal 1', 'Canal 2', 'Canal 3', 'Canal 4', 'Canal 5', 'Canal 6', 'Canal 7', 'Canal 8'], loc = 'best', fontsize="6")
plt.show()



# for gesto in range(1,7):#agrupa las varianzas por gesto y luego calcula el promedio simple
#     gesto_var = []
#     cont = 0
#     for i in range(vec_var.shape[0]):
#         if label[i] == gesto:
#             gesto_var.append(vec_var[i])
#             cont += 1
#     var_prom_dB1.append(sum(gesto_var)/cont) #promedio simple

# for gesto in range(1,7):#agrupa las varianzas por gesto y luego calcula el promedio simple
#     gesto_var = []
#     cont = 0
#     for i in range(varianzas_dB2.shape[0]):
#         if label_dB2[i] == gesto:
#             gesto_var.append(varianzas_dB2[i])
#             cont += 1
#     var_prom_dB2.append(sum(gesto_var)/cont) #promedio simple

# diff = abs(np.subtract(var_prom_dB1, var_prom_dB2, dtype=None))

