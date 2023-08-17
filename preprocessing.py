#Make preprocessing to signals for neural network
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# #function of normalizated min max
# def minmax_norm(df_input):
#     return (df_input - df_input.min()) / ( df_input.max() - df_input.min())*2 - 1
# # Create a dataframe with all the data without preprocessing

# data = pd.DataFrame()
# frames = []
# for i in range(1,73):
#     df = pd.read_csv("datashets/"+str(i)+".txt", sep="\t")
#     frames.append(df)

# data = pd.concat(frames)

# data.loc[data['class'] == 7 , ['class']] = 0

# data[['channel1','channel2','channel3','channel4','channel5','channel6','channel7','channel8']] = minmax_norm(data[['channel1','channel2','channel3','channel4','channel5','channel6','channel7','channel8']])
# data.loc[data['class'] == 0 , ['channel1','channel2','channel3','channel4','channel5','channel6','channel7','channel8']] = 0

# data.to_csv("dataframe_nopreprocesado.csv")

#dataframe normalizado y con gestos del 0 al 6 
data = pd.read_csv("dataframe_nopreprocesado.csv")
data = data.drop(["Unnamed: 0"], axis=1) #borra una fila que se crea cuando se lee el csv



print(data)
plt.figure()
plt.grid()
plt.plot(data['time'],data['channel5'])
plt.show()
plt.figure()
plt.grid()
plt.plot(data['time'],data['channel2'])
plt.show()


