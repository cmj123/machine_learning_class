# import libraries
import numpy as np

#List
L = [1,2,3]
A = np.array([1,2,3])

for e in L:
    print(e)

for e in A:
    print(e)

# Add information - list vs array
L.append(4)
print(L)

# A.append(4)
# print(A)

# another way for List
L = L +[5]
print(L)

# A = A + [4,5]
# print(A)

# List vs Array operations
L2 = []
for e in L:
    L2.append(e+e)
print(L2)

print(A+A)
print(2*A)

#
L2 = []
for e in L:
    L2.append(e*e)
print(L2)

print(A**A)

print(np.sqrt(A))

print(np.log(A))

# Evaluating loops vs matrix operations
a = np.array([1,2])
b = np.array([2,1])
dot = 0
for e, f in zip(a, b):
    dot+= e*f
print(dot)

print(np.sum(a*b))

print(np.dot(a,b))
