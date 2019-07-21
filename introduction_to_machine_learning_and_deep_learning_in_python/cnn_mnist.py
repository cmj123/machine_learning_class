# Import key libraries
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D

# we can load the MNIST dataset from keras datasets
# 60,000 training samples and 10,000images in test set
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("X_train original shape ", X_train.shape)
print("y_train original shape ", y_train.shape)
print("X_test original shape ", X_test.shape)
print("y_test original shape ", y_test.shape)
