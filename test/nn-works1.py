import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from test.scrap import sigmoid


def ReLU(Z):
    return np.maximum(Z,0)

def derivative_ReLU(Z):
    return Z > 0

def softmax(Z):
    """Compute softmax values for each sets of scores in x."""
    exp = np.exp(Z - np.max(Z)) #le np.max(Z) evite un overflow en diminuant le contenu de exp
    return exp / exp.sum(axis=0)


def init_params(size):
    np.random.seed(1)
    W1 = np.random.rand(10,size) - 0.5
    b1 = np.random.rand(10,1) - 0.5
    W2 = np.random.rand(10,10) - 0.5
    b2 = np.random.rand(10,1) - 0.5
    return W1,b1,W2,b2

def init_params_performance(size):
    np.random.seed(1)
    W1 = np.random.normal(size=(10, size)) * np.sqrt(1./(size))
    W2 = np.random.normal(size=(10, 10)) * np.sqrt(1./20)

    #b1 = np.random.normal(size=(10, 1)) * np.sqrt(1./10)
    #b2 = np.random.normal(size=(10, 1)) * np.sqrt(1./(size))
    ## definately better with the following
    b1 = np.zeros((10, 1))
    b2 = np.zeros((10, 1))

    return W1, b1, W2, b2

def init_params_new(size):
    np.random.seed(1)
    ## trying new between -1 to 1
    W1 = 2* np.random.random((10, size)) - 1
    b1 = 2* np.random.random((10, 1)) - 1
    W2 = 2* np.random.random((10, 10)) - 1
    b2 = 2* np.random.random((10, 1)) - 1

    return W1, b1, W2, b2

def forward_propagation(X,W1,b1,W2,b2):
    Z1 = W1.dot(X) + b1 #10, m 
    A1 = ReLU(Z1) # 10,m
    print("Shape W1:", W1.shape, "X1:", X.shape, "Z1:", Z1.shape, " b1:", b1.shape)
    Z2 = W2.dot(A1) + b2 #10,m
    print("Shape W2:", W2.shape, "A1:", A1.shape, "Z2:", Z2.shape, " b2:", b2.shape)
    A2 = softmax(Z2) #10,m
    return Z1, A1, Z2, A2

def one_hot(Y):
    ''' return an 0 vector with 1 only in the position correspondind to the value in Y'''
    one_hot_Y = np.zeros((Y.max()+1,Y.size)) #si le chiffre le plus grand dans Y est 9 ca fait 10 lignes
    one_hot_Y[Y,np.arange(Y.size)] = 1 # met un 1 en ligne Y[i] et en colonne i, change l'ordre mais pas le nombre
    return one_hot_Y

def backward_propagation(X, Y, A1, A2, W2, Z1, m):
    one_hot_Y = one_hot(Y)
    dZ2 = 2*(A2 - one_hot_Y) #10,m
    dW2 = 1/m * (dZ2.dot(A1.T)) # 10 , 10
    db2 = 1/m * np.sum(dZ2,1) # 10, 1
    dZ1 = W2.T.dot(dZ2)*derivative_ReLU(Z1) # 10, m
    dW1 = 1/m * (dZ1.dot(X.T)) #10, 784
    db1 = 1/m * np.sum(dZ1,1) # 10, 1

    return dW1, db1, dW2, db2

def update_params(alpha, W1, b1, W2, b2, dW1, db1, dW2, db2):
    W1 -= alpha * dW1
    b1 -= alpha * np.reshape(db1, (10,1))
    W2 -= alpha * dW2
    b2 -= alpha * np.reshape(db2, (10,1))

    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y)/Y.size

def gradient_descent(X, Y, alpha, iterations):
    size , m = X.shape

    ##W1, b1, W2, b2 = init_params(size)
    W1, b1, W2, b2 = init_params_performance(size)
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backward_propagation(X, Y, A1, A2, W2, Z1, m)

        W1, b1, W2, b2 = update_params(alpha, W1, b1, W2, b2, dW1, db1, dW2, db2)

        if (i+1) % int(iterations/10) == 0:
            print(f"Iteration: {i+1} / {iterations}")
            prediction = get_predictions(A2)
            print(f'{get_accuracy(prediction, Y):.3%}')
    return W1, b1, W2, b2

def make_predictions(X, W1 ,b1, W2, b2):
    _, _, _, A2 = forward_propagation(X, W1, b1, W2, b2)
    predictions = get_predictions(A2)
    return predictions

def show_prediction(index,X, Y, W1, b1, W2, b2):
    # None => cree un nouvel axe de dimension 1, cela a pour effet de transposer X[:,index] qui un np.array de dimension 1 (ligne) et qui devient un vecteur (colonne)
    #  ce qui correspond bien a ce qui est demande par make_predictions qui attend une matrice dont les colonnes sont les pixels de l'image, la on donne une seule colonne
    vect_X = X[:, index,None]
    prediction = make_predictions(vect_X, W1, b1, W2, b2)
    label = Y[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = vect_X.reshape((WIDTH, HEIGHT)) * SCALE_FACTOR

    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


############### MAIN #############################

print("Neural Net Work from train.csv.zip")
SCALE_FACTOR = 255

WIDTH = 28  #X_train.shape[1]
HEIGHT = 28     # X_train.shape[2]

print("X_Train Width:", WIDTH, " Height:", HEIGHT)

## data file play

#data = pd.read_csv('./train.csv')
data = pd.read_csv('../prototype/train.csv.zip',compression='zip')
print("Data head:")
print(data.head())

data = np.array(data)

m, n = data.shape
#print("Read Training Data :")
#print(data)
print("Data Shape:" , data.shape, " m:" ,m , " n:", n)

#np.random.shuffle(data) 

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / SCALE_FACTOR

print("X_Train shape:", X_train.shape )
print("Y_Train shape:", Y_train.shape , " shape[0]:", Y_train.shape[0])

size , m = X_train.shape

""""
np.random.seed(1)
W1 = np.random.normal(size=(30, size)) * np.sqrt(1./(size))
W2 = np.random.normal(size=(20, 30)) * np.sqrt(1./(30))
W3 = np.random.normal(size=(10, 20)) * np.sqrt(1./20)
b1 = np.zeros((30, 1))
b2 = np.zeros((20, 1))
b3 = np.zeros((10, 1))

print("W1:", W1.shape, "b1:", b1.shape)
print("W2:", W2.shape, "b2:", b2.shape)
print("W3:", W3.shape, "b3:", b3.shape)
#W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.15, 1)

print("******* FORWARD PROP *******")
X = X_train

## calulate 1
Z1 = W1.dot(X) + b1     #30,m 
#A1 = ReLU(Z1)           #30,m
A1 = sigmoid(Z1)        #30, m

print("m:", m)
print("N1 W1:", W1.shape,  "b1:", b1.shape, "Input X1:", X.shape, " ==>Z1(30,",m, "):", Z1.shape, "A1(30,", m, "):", A1.shape)

## calulate 2
Z2 = W2.dot(A1) + b2    #20,m 
#A2 = ReLU(Z2)           #20,m
A2= sigmoid(Z2)

print("N2 W2:", W2.shape,  "b2:", b2.shape, "Input A1:", A1.shape, " ==>Z2(20,",m,"):", Z2.shape, "A2(20,",m,"):", A2.shape)

Z3 = W3.dot(A2) + b3    #10,m
#A3 = softmax(Z3)        #10,m
A3=sigmoid(Z3)

print("N3 W3:", W3.shape,  "b3:", b3.shape, "Input A2:", A2.shape, " ==>Z3(10,",m,"):", Z3.shape, "A3(10,",m,"):", A3.shape)


print("******** BACK PROP********")
Y = Y_train
one_hot_Y = one_hot(Y)

#calculate 3
dZ3 = 2*(A3 - one_hot_Y)        #10,m
dW3 = 1/m * (dZ3.dot(A2.T))     # 10, 20
db3 = 1/m * np.sum(dZ3,1)       # 10, 1

print("N3 dZ3(10,m):", dZ3.shape, "dW3(10,20):", dW3.shape, "db(10,)3:", db3.shape)

print("Z2:", Z2.shape , " W2:", W2.shape)

dZ2 = W2.T.dot(dZ3)*derivative_ReLU(Z2)     # 20, m
dW2 = 1/m * (dZ2.dot(A1.T))                 #10, 20
db2 = 1/m * np.sum(dZ2,1)                   # 20, 1

print("N2 dZ2(10,m):", dZ2.shape, "dW2(10,20):", dW2.shape, "db(20,)3:", db2.shape)

"""

W1, b1, W2, b2 = init_params_performance(size)

print("******* FORWARD PROP *******")
X = X_train
Z1 = W1.dot(X) + b1     #10,m 
A1 = ReLU(Z1)           #10,m

Z2 = W2.dot(A1) + b2    #10,m
A2 = softmax(Z2)        #10,m


print("N1 W1:", W1.shape, "X1:", X.shape, "b1:", b1.shape, "Z1:", Z1.shape, "A1:", A1.shape)
print("N3 W3:", W2.shape, "A1:", A1.shape, "b3:", b2.shape,  "Z3:", Z2.shape, " A3:", A2.shape)

print("******** BACK PROP********")
Y = Y_train
one_hot_Y = one_hot(Y)
dZ2 = 2*(A2 - one_hot_Y) #10,m
dW2 = 1/m * (dZ2.dot(A1.T)) # 10 , 10
db2 = 1/m * np.sum(dZ2,1) # 10, 1

dZ1 = W2.T.dot(dZ2)*derivative_ReLU(Z1) # 10, m
dW1 = 1/m * (dZ1.dot(X.T)) #10, 784
db1 = 1/m * np.sum(dZ1,1) # 10, 1

print("N3 dZ3:", dZ2.shape, "dW3:", dW2.shape, "db3:", db2.shape)
print("dZ3 <== 2*(A3 - one_hot_Y) Y:", Y.shape, "A3:", A2.shape , "-HotY:", one_hot_Y.shape)
print("dW3 <== 1/m * (dZ3.dot(A1.T)): dZ3.dot:", dZ2.shape, " A1.T:", A1.T.shape)
print("db3 <== 1/m * np.sum(dZ3,1) np.sum(dZ3,1):", dZ2.shape, "db2:", db2.shape)

print("N1 dZ1:", dZ1.shape, "dW1:", dW1.shape, "db1:", db1.shape)
print("dZ2 <==W2.T.dot(dZ2)*derivative_ReLU(Z1)  W2.T.dot:", W2.T.shape, "(dZ2)", dZ2.shape , "*deriv(Z1):", derivative_ReLU(Z1).shape)
print("dW1 <== 1/m * (dZ1.dot(X.T)) dZ1.dot:", dZ1.shape, "(X.T):", X.T.shape)
print("db1 <== 1/m * np.sum(dZ1,1): np.sum(dZ1,1):", dZ1.shape)

print("np.sum(dZ2):", np.sum(dZ2))


