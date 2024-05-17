import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

size = (28,28)

dataframe = pd.read_csv('database_reduc_rms.csv') #lee el dataframe


dataframe = dataframe.to_numpy()
filas, columnas = dataframe.shape
label = []
clase = 0
list_list = []
for i in range(filas):
    if clase == dataframe[i,8]:
        list.append(dataframe[i,:7])
    else:
        list = []
        clase = dataframe[i,8]
        label.append(clase)
        list_list.append(list)
    

aux_np = np.array(list_list[0])

aux_var = np.zeros(7)
vec_var = np.zeros((1,7))


for i in range(len(list_list)):
    var_aux = np.array(list_list[i])
    for j in range(7):
        aux_var[j] = var_aux[:,j].var()
    vec_var = np.insert(vec_var, vec_var.shape[0], aux_var, 0)

vec_var = np.delete(vec_var, 0, axis= 0)
print(vec_var.shape)
print(vec_var)

plt.figure()
plt.plot(label,vec_var[:,0], 'b.')
plt.grid()
plt.show()



# dB = []
# for i in range(0,len(list_list)):
#     aux = np.zeros((1,7))
#     aux = np.delete(aux, 0, axis=0)
#     for l in list_list[i]:
#         aux = np.insert(aux, aux.shape[0] , l, 0)  
#     aux = cv2.resize(aux, dsize=(28, 28), interpolation=cv2.INTER_LINEAR)
#     dB.append(aux)  

# dB = np.array(dB)
# # tamaño de dB (864,28,28)  hay 864 "imagenes" de 28x28

# #array de etiquetas
# label = np.array(label, dtype=int)

# #imprime los tamaños de cada arreglo
# print(label.shape)
# print(dB.shape)


# #grafica el vector de etiquetas
# plt.figure()
# plt.grid()
# plt.plot(label)
# plt.show()

# #muestra 30 imagenes
# for fig in range(0,30):
#     plt.subplot(5,6,fig+1)
#     plt.title(label[fig])
#     plt.imshow(dB[fig],cmap=plt.cm.binary)

# plt.show()


# np.savetxt("labels_base1_gestos.csv", label, delimiter=",")

# #guarda las imagenes
# for i in range(864):
#     im = np.array(dB[i] * 255, dtype = np.uint8)
#     threshed = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
#     cv2.imwrite("img_datashet/"+str(i)+".png", threshed)


# np.save("Base1_gestos_separados.npy", dB)








