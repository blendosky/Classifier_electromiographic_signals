import numpy as np
import matplotlib.pyplot as plt


from sklearn.preprocessing import MinMaxScaler


datos = []
with open('datashets/2.txt') as fname:
	lineas = fname.readlines()
	for linea in lineas:
		datos.append(linea.strip('\n'))

datos.pop(0)

database=[]

for prof in datos:
	database.append(prof.split())
	
dat = []
for x in database:
	sublist = []
	for xin in x:
		sublist.append(float(xin))
	dat.append(sublist)



dat = np.array(dat)
#print(dat[:10,:2])#database complete ok, its name is dat

X = dat[:,:8]
y = dat[:,9]

print(X)
print(y)


#print(database)	



