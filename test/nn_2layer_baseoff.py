import pandas as pd
import numpy as np
import pickle
from keras.datasets import mnist
import matplotlib.pyplot as plt

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

def forward_propagation(X,W1,b1,W2,b2):
    Z1 = W1.dot(X) + b1 #10, m
    A1 = ReLU(Z1) # 10,m
    Z2 = W2.dot(A1) + b2 #10,m
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

    W1, b1, W2, b2 = init_params_performance(size)
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backward_propagation(X, Y, A1, A2, W2, Z1, m)

        W1, b1, W2, b2 = update_params(alpha, W1, b1, W2, b2, dW1, db1, dW2, db2)

        if (i+1) % (iterations/10) == 0:
            print(f"Iteration: {i+1} / {iterations}")
            prediction = get_predictions(A2)
            print(f'{get_accuracy(prediction, Y):.3%}')
    return W1, b1, W2, b2

def make_predictions(X, W1 ,b1, W2, b2):
    _, _, _, A2 = forward_propagation(X, W1, b1, W2, b2)
    predictions = get_predictions(A2)
    return predictions

def show_prediction(index,X, Y, W1, b1, W2, b2, WIDTH, HEIGHT, SCALE_FACTOR):
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

def read_data_from_zip():
    print("Neural Net Work from train.csv.zip")
    SCALE_FACTOR = 255
    WIDTH = 28  #X_train.shape[1]
    HEIGHT = 28     # X_train.shape[2]
    print("X_Train Width:", WIDTH, " Height:", HEIGHT)

    data = pd.read_csv('../prototype/train.csv.zip',compression='zip')
    print("Data head:")
    print(data.head())

    data = np.array(data)
    m, n = data.shape
    #print("Read Training Data :")
    #print(data)
    print("Data Shape:" , data.shape, " m:" ,m , " n:", n)

    data_test = data[0:1000].T
    Y_test = data_test[0]
    X_test = data_test[1:n]
    X_test = X_test / SCALE_FACTOR

    #np.random.shuffle(data) 
    
    data_train = data[1000:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n]
    X_train = X_train / SCALE_FACTOR

    print("X_Train shape:", X_train.shape )
    print("Y_Train shape:", Y_train.shape , " shape[0]:", Y_train.shape[0])
    
    return X_train, Y_train, X_test, Y_test, WIDTH, HEIGHT, SCALE_FACTOR


def read_data_mnist():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    SCALE_FACTOR = 255 # TRES IMPORTANT SINON OVERFLOW SUR EXP
    WIDTH = X_train.shape[1]
    HEIGHT = X_train.shape[2]


    X_train = X_train.reshape(X_train.shape[0],WIDTH*HEIGHT).T / SCALE_FACTOR
    X_test = X_test.reshape(X_test.shape[0],WIDTH*HEIGHT).T  / SCALE_FACTOR
    
    return X_train, Y_train, X_test, Y_test, WIDTH, HEIGHT, SCALE_FACTOR
    
############## MAIN ##############

#X_train, Y_train, X_test, Y_test , WIDTH, HEIGHT, SCALE_FACTOR = read_data_mnist()
X_train, Y_train, X_test, Y_test, WIDTH, HEIGHT, SCALE_FACTOR = read_data_from_zip()

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.15, 200)

with open("trained_params.pkl","wb") as dump_file:
    pickle.dump((W1, b1, W2, b2),dump_file)

with open("trained_params.pkl","rb") as dump_file:
    W1, b1, W2, b2=pickle.load(dump_file)
show_prediction(0,X_test, Y_test, W1, b1, W2, b2, WIDTH, HEIGHT, SCALE_FACTOR)
show_prediction(1,X_test, Y_test, W1, b1, W2, b2, WIDTH, HEIGHT, SCALE_FACTOR)
show_prediction(2,X_test, Y_test, W1, b1, W2, b2, WIDTH, HEIGHT, SCALE_FACTOR)
show_prediction(100,X_test, Y_test, W1, b1, W2, b2, WIDTH, HEIGHT, SCALE_FACTOR)
show_prediction(200,X_test, Y_test, W1, b1, W2, b2, WIDTH, HEIGHT, SCALE_FACTOR)



