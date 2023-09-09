import numpy as np
import matplotlib.pyplot as plt

# IMAGEN GRIS OSCURO
# Una matriz con valor 0.2 en todos sus elementos 
size=(20,30)

imagen_gris_oscuro = np.ones(size)*0.5

#visualizamos la matriz
#Se ve como una imagen gris, ya que todos los elementos (pixeles) tienen intensidad 0.5
plt.imshow(imagen_gris_oscuro,vmin=0,vmax=1)

#creamos otra figura para mostrar la imagen (sino el proximo imshow sobreescribe al anterior)
plt.figure()

# IMAGEN ALEATORIA
# Una matriz con valor aleatorio
imagen_aleatoria = np.random.rand(size[0],size[1])
print(imagen_aleatoria)
#visualizamos la matriz
#Se ve como una imagen gris, ya que todos los elementos (pixeles) tienen intensidad 0.5
plt.imshow(imagen_aleatoria,vmin=0,vmax=1)
plt.show()