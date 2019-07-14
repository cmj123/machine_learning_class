# Import libraries
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

# Create dataset
x,y = make_blobs(n_samples=100, centers=5,random_state=0,cluster_std=2)
plt.scatter(x[:,0],x[:,1],s=75)
plt.show()

# create model - define, fit and predcict
est = KMeans(n_clusters=3)
est.fit(x)
y_kmeans = est.predict(x)

# Visualise results
plt.scatter(x[:,0],x[:,1],c=y_kmeans, s=75, cmap='rainbow')
plt.show()
