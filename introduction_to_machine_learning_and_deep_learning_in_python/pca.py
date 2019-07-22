# Import the libraries
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
import matplotlib.cm as cm

# 8x8 pixel per image - > 64 features !!! Humans are not able to cope with this
# This is why we use PCA -> reduce the dimensions: we can visualise the data in 2D
# We want the investiagte if the distribution after PCA reveals the
# distribution of different classes, and if they are clearly separable


digits = load_digits()
X_digits, y_digits = digits.data, digits.target

images_and_labels = list(zip(digits.images, digits.target))

for index, (image, label) in enumerate(images_and_labels[7:13]):
    plt.subplot(2,3, index+1)
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Target: %i' % label)

plt.show()

# Build the pca model
estimator = PCA(n_components=10)
X_pca = estimator.fit_transform(X_digits)

colors = ['black','blue','purple','yellow','white','red','lime','cyan','orange','gray']

for i in range(len(colors)):
    px = X_pca[:, 0][y_digits == i]
    py = X_pca[:, 1][y_digits == i]
    plt.scatter(px,py, c=colors[i])
    plt.legend(digits.target_names)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')

plt.show()
