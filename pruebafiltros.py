import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter, freqz
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def minmax_norm(df_input):
    return (df_input - df_input.min()) / ( df_input.max() - df_input.min())*2 - 1
# Create a dataframe with all the data without preprocessing


#se filtra la señal de entrada y se crea una base de datos con la señal filtrada

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


frames = []
for i in range(1,73):
    df = pd.read_csv("datashets/"+str(i)+".txt", sep="\t")
    frames.append(df)

data = pd.concat(frames)

data.drop(['time'], axis=1, inplace=True)

data.reset_index(inplace=True, drop=True)

data[['channel1','channel2','channel3','channel4','channel5','channel6','channel7','channel8']] = data[['channel1','channel2','channel3','channel4','channel5','channel6','channel7','channel8']] * 10000
data = round(data,2)

order = 10
fs = 1000
cutoff = 100


X = data[['channel1','channel2','channel3','channel4','channel5','channel6','channel7','channel8']]
Y = data['class']


X_fil = []
for i in range(1,9):
    X_fil.append(butter_lowpass_filter(data['channel'+str(i)+''], cutoff, fs, order))



df = pd.DataFrame((zip(X_fil[0],X_fil[1],X_fil[2],X_fil[3],X_fil[4],X_fil[5],X_fil[6],X_fil[7])), 
                  columns = ['channel1','channel2','channel3','channel4','channel5','channel6','channel7','channel8'])
df.reset_index(inplace=True, drop=True)
df = round(df,1)

plt.subplot(2,1,1)
plt.plot(abs(df['channel1']))
plt.title('Filtrado')
plt.plot(data['class'])
plt.grid()
plt.subplot(2,1,2)
plt.plot(abs(data['channel1']))
plt.title('Sin filtrado')
plt.plot(data['class'])
plt.grid()
plt.show()

data = df
data['class'] = Y
data.reset_index(inplace=True, drop=True)

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

#plotea el canal con rms, comparandolo con la clase --> se redujo el tamaño de toda la base de datos a 29980
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
data.reset_index(inplace=True, drop=True)


#guardar el dataframe
data_comp.to_csv('database_reduc_filter.csv', header=True, index=False)
print(data_comp.shape)


# X_train, X_test, y_train, y_test = train_test_split(X_fil, Y, test_size=0.2)
# y_train = y_train.astype('int')
# y_test = y_test.astype('int')




# model = MLPClassifier(hidden_layer_sizes=(50,25,12), max_iter=600, alpha=1e-7,
#                     solver='adam', verbose=True, random_state=1,
#                     learning_rate_init=.01, tol=1e-5, n_iter_no_change=100)

# model.fit(X_train, y_train)


# y_predict = model.predict(X_test)

# color = np.where(y_predict != y_test, "green", "red")


# fig, ax = plt.subplots()
# ax.scatter(y_predict, y_test, c = color, alpha=0.01)
# plt.show()


# precision = accuracy_score(y_test, y_predict)

# print(precision)
