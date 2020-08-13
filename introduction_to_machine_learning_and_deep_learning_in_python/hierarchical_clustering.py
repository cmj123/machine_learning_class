# Import key libraries
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

# Create dataset
X = np.array([[1,1],[1.5, 1],[3,3],[4,4],[3,3.5],[3.5,4]])
# plt.scatter(X[:,0],X[:,1])
# plt.show()

# "single" -> nearest point generator algorithm d(u, v) = min(dist(point_i, point_j))
linkage_matrix = linkage(X, "single")

print(linkage_matrix)

# Pruning the tree like structure with truncate mode
dendrogram = dendrogram(linkage_matrix,truncate_mode="none")

plt.title("Hierarchical Clustering")

plt.show()
