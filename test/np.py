import numpy as np
import json

def ReLU(Z):
    return np.maximum(Z,0)
"""
npa = np.array([1,2, 3])
npb = np.array([1,2,4])
npd = np.array([1, 3, 5])

npe = np.array([2,8])
npf = np.array([2,9])
a = []
print("a:", a, " len:", len(a))
print("npa:", npa)
a.append([npa])
print("a:", a, " len:", len(a), " test i:" , (0 in a))
e = a[0]
print("e=", e)
if e is not None:
    print("is not none:")
e.append(npb)
print("a:", a, " len:", len(a))
"""

t = np.array([[0, 0, 1],[1, 1, 1],[1, 0, 1],[0, 1, 1] ]).T

np.random.seed(1)
w1= np.random.rand(2,3) + 1.0;  # 2 sets of 3 input node 
b1 = np.zeros((2, 1))
w2 = np.random.rand(1, 2) +1.0  # 1 set of 2 input
b2 = np.zeros((1,1)) + 1.0

print("w1:", w1)
print("Shape w1:", w1.shape, "t:", t.shape, "b1:", b1.shape)
z1=w1.dot(t)+b1
print("Shape z1:",z1.shape, " z1:", z1 )
a1 = ReLU(z1)
print("Shape a1:",a1.shape, " a1:", a1 )
print("Shape w2:", w2.shape, " a1:", a1.shape, " b2:", b2.shape)
z2 = w2.dot(a1) + b2
print("Shape z2:", z2.shape, " z2:", z2)
#print("w:", w, " t(w):", w.T)
#print( w.dot(t))
#print(np.dot(t[0], w.T))
#print(np.dot(t[2], w))
#print(np.dot(t[2], w.T))

