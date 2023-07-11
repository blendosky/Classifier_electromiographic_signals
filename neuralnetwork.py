import pickle
import matplotlib.pyplot as plt
import numpy as np


with open("obj.pickle", "rb") as f:
    database = pickle.load(f)
# Imprime [1, 2, 3, 4]

print(len(database))
database_red = []
cont=0
for fil in database:#reduzco los datos a la mitad
    if cont>=2:
        cont=0
        database_red.append(fil)
    cont += 1

print(len(database_red))

X = database[0:10]
print(X)

print()
# plt.figure()
# plt.scatter(x[:50000],)

#tomar conjuntos de 20 datos y calcular el rms

# fila = []
# col = []
# for fil in database:
#     fila.append(fil[0])
#     col.append(fil[9])
# fila.pop(0)
# col.pop(0)
# plt.plot(fila,col)
# plt.show()
