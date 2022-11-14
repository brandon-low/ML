
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import pickle
#from keras.datasets import mnist


data = pd.read_csv('/home/brandon/Downloads/tmp/Wokspace-MachineLearning/train.csv')

print("Data head:")
print(data.head())

data = np.array(data)
print("Read Training Data :")
print(data)

m, n = data.shape       # m is row n is column
np.random.shuffle(data) # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T   #use all the data and transpose it from row toto column
Y_dev = data_dev[0]     # is the 1st row
X_dev = data_dev[1:n]   # the rest
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
#_,m_train = X_train.shape

print("Y_train:")
print(Y_train)

#print("X_train shape at 0:", X_train[0].shape)
#print("X_train shape at :", X_train[:, 0].shape) #should put out 784 pixel

#synaptic_weights = 2 * np.random.random((3, 1)) -1
#print("Generate random number:", 2* np.random.random((10, 10)) - 1 )
## randn output -0.5 to 0.5 so subtraction was unneeded + breaks everything
def init_params():
    ##Original
    W1 = np.random.randn(10, 784) - 0.5
    b1 = np.random.randn(10, 1) - 0.5
    W2 = np.random.randn(10, 10) -0.5
    b2 = np.random.randn(10, 1) -0.5
    return W1, b1, W2, b2

def init_params_performance():
    W1 = np.random.normal(size=(10, 784)) * np.sqrt(1./(784))
    W2 = np.random.normal(size=(10, 10)) * np.sqrt(1./20)

    #b1 = np.random.normal(size=(10, 1)) * np.sqrt(1./10)
    #b2 = np.random.normal(size=(10, 1)) * np.sqrt(1./(784))
    b1 = np.zeros((10, 1))
    b2 = np.zeros((10, 1))
    return W1, b1, W2, b2

def init_params_new():
    ## trying new between -1 to 1
    W1 = 2* np.random.random((10, 784)) - 1
    b1 = 2* np.random.random((10, 1)) - 1
    W2 = 2* np.random.random((10, 10)) - 1
    b2 = 2* np.random.random((10, 1)) - 1

    return W1, b1, W2, b2


def ReLU(Z):
    return np.maximum(Z, 0)
    #return np.maximum(Z, 0)

def ReLU_deriv(Z):
    return Z > 0

def softmax(Z):
     """Compute softmax values for each sets of scores in x."""
    exp = np.exp(Z - np.max(Z)) #le np.max(Z) evite un overflow en diminuant le contenu de exp
    return exp / exp.sum(axis=0)
    #A = np.exp(Z) / sum(np.exp(Z))
    #return A

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
     #old stufff

    #W1 = W1 - alpha * dW1
    #b1 = b1 - alpha * db1
    #W2 = W2 - alpha * dW2
    #b2 = b2 - alpha * db2

    ## need to reshape db1 & db2
    #db2 = 1/m * np.sum(dZ2,1) # 10, 1
    #db1 = 1/m * np.sum(dZ1,1) # 10, 1

    W1 -= alpha * dW1
    b1 -= alpha * np.reshape(db1, (10,1))
    W2 -= alpha * dW2
    b2 -= alpha * np.reshape(db2, (10,1))

    return W1, b1, W2, b2


def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()

    print("Weight 1:", W1)
    print("Bias 1:" , b1)
    print("Weight 2:", W2)
    print("Bias 2:" , b2)

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            print("Adjustments: ", dW1, db1, dW2, db2)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

#test init_params
W1, b1, W2, b2 = init_params_performance()
print("Weight 1:", W1)
print("Bias 1:" , b1)
print("Weight 2:", W2)
print("Bias 2:" , b2)

# test gradient_descent
#W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 500, 0.10)





