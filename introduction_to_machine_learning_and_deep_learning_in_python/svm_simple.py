import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm

# create dataset - red and blue class
# Blue class
x1 = np.array([0.3,0.5,1,1.4,1.7,2])
y1 = np.array([1,4.5,2.3,1.9,8.4,2])

#Red class
x2 = np.array([3.3,3.5,4,4.4,5.7,6])
y2 = np.array([7,1.5,6.3,1.9,2.9,7.1])

X = np.array([[0.3,1],[0.5,4.5],[1,2.3],[1.4,1.9],[1.7,8.9],[2,4.1],[3.3,7],[3.5,1.5],[4,6.3],[4.4,1.9],[5.7,2.9],[6,7.1]])
y = np.array([0,0,0,0,0,0,1,1,1,1,1,1]) # 0: blue class, 1: red class


plt.plot(x1, y1, 'ro', color='blue')
plt.plot(x2, y2, 'ro', color='red')
plt.plot(3,5,'ro', color='green')

classifier = svm.SVC()
classifier .fit(X,y)
print(classifier .predict([[3,5]]))

plt.show()
