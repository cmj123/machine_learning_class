# Import key libraries
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
import matplotlib.cm as cm

# Load digits
digits = load_digits()
print(digits)
