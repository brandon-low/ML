import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from keras.datasets import mnist

print("Hello Play")
SCALE_FACTOR = 255

"""
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

WIDTH = X_train.shape[1]
HEIGHT = X_train.shape[2]

print("X_Train Width:", WIDTH, " Height:", HEIGHT)

print("X_Train shape:", X_train.shape )
print("Y_Train shape:", Y_train.shape )

print("X_test shape:", X_test.shape )
print("Y_test shape:", Y_test.shape )

X_train = X_train.reshape(X_train.shape[0],WIDTH*HEIGHT).T / SCALE_FACTOR
X_test = X_test.reshape(X_test.shape[0],WIDTH*HEIGHT).T / SCALE_FACTOR

index = 0
vect_X = X_test[:, index,None]
current_image = vect_X.reshape((WIDTH, HEIGHT)) * SCALE_FACTOR

plt.gray()
plt.imshow(current_image, interpolation='nearest')
plt.show()

print("label:" , Y_test[index])
"""

## data file play
data = pd.read_csv('/home/brandon/Downloads/tmp/Wokspace-MachineLearning/train.csv')
#print("Data head:")
#print(data.head())

data = np.array(data)
m, n = data.shape
#print("Read Training Data :")
#print(data)
print("Data Shape " , data.shape)


data_train = data[1000:m].T
Y_train2 = data_train[0]
X_train2 = data_train[1:n]
X_train2 = X_train2 / SCALE_FACTOR
print("X_Train2 shape:", X_train2.shape )
print("Y_Train2 shape:", Y_train2.shape )
x2m, x2n = X_train2.shape
print("X train 2 Shape m:", x2m, " n:" , x2n)

index = 11

print("Y train 2 Label:", Y_train2[index])

vect_X2 = X_train2[:, index ,None]
image2 = vect_X2.reshape((28, 28))

plt.gray()
plt.imshow(image2, interpolation='nearest')
plt.show()



