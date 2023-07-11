import matplotlib.pyplot as plt
import numpy as np


from matplotlib.colors import ListedColormap

from sklearn import datasets
from sklearn.neural_network import MLPClassifier




iris = datasets.load_iris()

X = iris.data[:, :3]  # we only take the first two features.
y = iris.target

print(X[:10,:2])

fig, axs = plt.subplots(2)
fig.suptitle("compare y datashets and y_classifier")
axs[0].scatter(X[:,0],X[:,1],c=y)
axs[0].set(xlabel='length sepal')
axs[0].set(ylabel='width sepal')

cm_bright = ListedColormap(["#FF0000", "#0000FF", "#00FF00"])

clf = MLPClassifier(solver='adam', alpha=1e-5,
                    hidden_layer_sizes=(50,20), random_state=1)

clf.fit(X,y)

y_classifier = clf.predict(X)


print(y_classifier)
print(y)
axs[1].scatter(X[:,0],X[:,1],c=y_classifier, cmap=cm_bright, edgecolors="k")
plt.show()





