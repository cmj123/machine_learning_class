# Import libraries
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn import datasets
import numpy as np

X,y = datasets.make_moons(n_samples=1500, noise=.05)
# print(X.shape)
# print(y.shape)
x1 = X[:,0]
x2 = X[:,1]

print("This is the dataset we want to classify with DBSCAN")
plt.scatter(x1,x2,s=5)
plt.show()

# results with DBSCAN algorithms
dbscan = DBSCAN(eps=0.1)
dbscan.fit(X)
y_pred = dbscan.labels_.astype(np.int)

colors = np.array(['#ff0000','#00ff00'])
print("These are the clusters with DBSCAN")
plt.scatter(x1, x2, s=5, color=colors[y_pred])
plt.show()

# results with k-means clustering
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
y_pred = kmeans.labels_.astype(np.int)

colors = np.array(['#ff0000','#00ff00'])
print("These are the clusters with KMeans")
plt.scatter(x1, x2, s=5, color=colors[y_pred])
plt.show()
