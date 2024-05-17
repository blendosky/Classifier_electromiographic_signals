import pandas as pd
import matplotlib.pyplot as plt



dataframe = pd.read_csv('database_reduc_rms.csv')

data = pd.read_csv('database.csv')

print("Las dimensiones de la base de datos sin preprocesar son:",data.shape)
print("Las dimensiones de la base de datos preprocesada son:",dataframe.shape)

plt.figure()
plt.grid()
plt.plot(dataframe['class'])
plt.plot(data['class'])
plt.show()