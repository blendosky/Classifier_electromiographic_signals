#Make preprocessing to signals for neural network
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm
import numpy as np

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

# data.reset_index(inplace=True, drop=True)

# data.loc[data['class'] == 7 , ['class']] = 0

# data[['channel1','channel2','channel3','channel4','channel5','channel6','channel7','channel8']] = minmax_norm(data[['channel1','channel2','channel3','channel4','channel5','channel6','channel7','channel8']])
# data.loc[data['class'] == 0 , ['channel1','channel2','channel3','channel4','channel5','channel6','channel7','channel8']] = 0

# data_class = data['class']

# data.drop(data_class.index[data_class == 0], axis=0, inplace=True)
       
# data.reset_index(inplace=True, drop=True)

# data.to_csv("dataframe_nopreprocesado.csv")

#dataframe normalizado y con gestos del 0 al 6 
data = pd.read_csv("dataframe_nopreprocesado.csv")
data = data.drop(["Unnamed: 0"], axis=1) #borra una fila que se crea cuando se lee el csv

data_figure = data

plt.figure(2)
plt.plot(data_figure['channel1'])
plt.grid()
plt.show()



fig, ax = plt.subplots(2,1)
ax[0].hist(data_figure["channel1"])
ax[0].grid()
ax[0].hist(data_figure["channel2"])
ax[0].hist(data_figure["channel3"])
ax[0].hist(data_figure["channel4"])
ax[0].hist(data_figure["channel5"])
ax[0].hist(data_figure["channel6"])
ax[0].hist(data_figure["channel7"])
ax[0].hist(data_figure["channel8"])
ax[1].hist(data_figure['class'])
ax[1].grid()
plt.show()

f = np.fft.fft(data_figure["channel1"])
freq = np.fft.fftfreq(len(f), d=1)

plt.figure(3)
plt.grid()
plt.plot(freq,f)
plt.show()
