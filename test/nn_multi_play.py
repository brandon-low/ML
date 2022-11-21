import numpy as np
import pandas as pd
import pickle
from keras.datasets import mnist
import matplotlib.pyplot as plt


# tanh activation function
def tanh(x: np.ndarray) -> np.ndarray:
    # return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    return np.tanh(x)

# Derivative of Tanh Activation Function
def derivative_tanh(x: np.ndarray) -> np.ndarray:
    #return 1 - np.power(tanh(z), 2)
    return 1 - np.power(activation(x), 2)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Sigmoid derivative for backpropagation
def derivative_sigmoid(x):
    #activation(x)*(1 - activation(x))
    return x*(1-x)

# Leaky_ReLU activation function
def leakyrelu(z, alpha):
    #return max(alpha * z, z)
    return np.max(alpha*z, z)
    
# Derivative of leaky_ReLU Activation Function
def dericative_leakyrelu(z, alpha):
    return 1 if z > 0 else alpha   

def ReLU(Z):
    # return max(0, z)
    return np.maximum(Z,0)

def derivative_ReLU(Z):
    #return 1 if z > 0 else 0
    return Z > 0

#Use for last layer
def softmax(Z):
    """Compute softmax values for each sets of scores in x."""
    exp = np.exp(Z - np.max(Z)) #le np.max(Z) evite un overflow en diminuant le contenu de exp
    return exp / exp.sum(axis=0)


def activation(x: np.ndarray) -> np.ndarray:
    # x = [-1*num for num in x]
    # return (1)/(1 + np.exp(x))
    #return np.tanh(x)
    #return tanh(x)
    return ReLU(x)
    #return sigmoid(x)

def activation_deriv(x: np.ndarray) -> np.ndarray:
    # return activation(x)*(1 - activation(x))
    #return 1 - np.power(activation(x), 2)
    #return derivative_tanh(x)
    return derivative_ReLU(x)
    #return derivative_sigmoid(x)

def one_hot(Y):
    ''' return an 0 vector with 1 only in the position correspondind to the value in Y'''
    one_hot_Y = np.zeros((Y.max()+1,Y.size)) #si le chiffre le plus grand dans Y est 9 ca fait 10 lignes
    one_hot_Y[Y,np.arange(Y.size)] = 1 # met un 1 en ligne Y[i] et en colonne i, change l'ordre mais pas le nombre
    return one_hot_Y

def init_params(cfg):
    weights = []
    biases = []
    np.random.seed(1)
    for i in range(len(cfg) -1):
        #print(nn_cfg[i+1] ,":", nn_cfg[i])
        #w.append(np.random.randn( cfg[i+1], cfg[i])) # dont use this 
         
        ## this is still the best
        weights.append(np.random.normal(size=(cfg[i+1], cfg[i])) * np.sqrt(1./(cfg[i])) )
        biases.append(np.zeros((cfg[i+1], 1)))
        
        if (i == 0):
            print("W:", weights[i].shape,"B:",biases[i].shape, "Input Layer")
        elif (i == (len(cfg)-2) ):
            print("W:", weights[i].shape, "B:",biases[i].shape, "Output Layer")
        else:
            print("W:", weights[i].shape, "B:",biases[i].shape, "Hidden Layer")
            
    
    return weights, biases
    
def forwardfeed(x: np.ndarray, weights, biases):
    zetas = []      # zetas
    acts = []      # activated
    
    # set the 1st entry
    zetas.append(x)
    acts.append(x)
    
    nWeights=len(weights)
    
    #print("**** FORWARD FEED  return the last activation which is the result ******")
    for i in range( len(weights) ):
        zetas.append(weights[i].dot(acts[i]) + biases[i])
        #print("z[-1]:", z[-1].shape , "i:", i, "z[",(i+1),"]", z[i+1].shape, "z[",i,"]", z[i].shape)
        if (i == (nWeights-1) ):
            #print("Forward Output Layer W:", w[i].shape)
            acts.append(softmax(zetas[i+1]))
        else:
            acts.append(activation(zetas[i+1]))
        
        #print("ForwardFeed i:", i, "z[-1]:", z[-1].shape, "a[-1]:", a[-1].shape)
    
    #print("return at forward a[-1]:", a[-1].shape)
    #print("**** FORWARD FEED FINISH******")
    return zetas, acts, acts[-1]

def backward_propagate(weights, biases, zetas, acts, Y):
    #print("**** BACK PROP  START ****")
    gradw = []
    gradb = []
    deltas = []
    
    #print("delta:", a[-1].shape, "-", Y_train.shape , "=", (a[-1] - Y_train).shape )
    one_hot_Y = one_hot(Y)
    deltas.append( 2 * (acts[-1] - one_hot_Y))    #dZ2 = 2*(A2 - one_hot_Y) #10,m
    
    #print("a[-2]:", a[-2].shape, " a[-2].T:", a[-2].T.shape, "deltas[-1]:", deltas[-1].shape)
    #print("DoLast gradw: " , deltas[-1].dot(a[-2].T).shape)

    total_m = deltas[-1].shape[1]
    #print("0 deltas to work out gradw & gradb: m:", total_m)
    mm = 1/total_m      ## same number for all
    
    #dW2 = 1/m * (dZ2.dot(A1.T)) # 10 , 10
    #db2 = 1/m * np.sum(dZ2,1) # 10, 1

    gradw.append(mm * deltas[-1].dot(acts[-2].T) ) # delta[out neurons, m]
    gradb.append(mm * np.sum(deltas[-1], 1)) # 1/mX sum(deltas)

    #gradb.append(np.sum(deltas[-1], axis=0, keepdims=True))
    #print("Start Loop Range deltas:" , len(deltas), " w:", len(w), "z:", len(z))
    #print("gradw:", gradw[-1].shape, "gradb:", gradb[-1].shape)
    #print("Size a:", len(a) , "a[-1]", a[-1].shape, "a[", (len(a)-1), "]:", a[len(a)-1].shape,
    #    "a[",  (len(a)-2), "]:", a[len(a)-2].shape)

    #print("***** BUILD Rest Of Deltas , gradw, gradb ******")
    #for i in range(nn_cfg_n_layers - 1, 0, -1):     # loop is wrong should stop at 1
    nWeights=len(weights)
    for i in range(nWeights, 1, -1):     # loop is right 
        #print("DO:", i);
        ## should use w [i-1]
        #print("z[",(i-1), "]", z[i-1].shape, "w[",i,"]", w[i-1].shape)
        #print("activate_dev:", activation_deriv(z[i]).shape)
        #print("w[i-1].T:", w[i-1].T.shape)
        #print("deltas[-1]:", deltas[-1].shape)
        #print("dZ=", w[i-1].T.dot(deltas[-1]).shape)
        #dZ1 = W2.T.dot(dZ2)*derivative_ReLU(Z1) # 10, m
        deltas.append( weights[i-1].T.dot(deltas[-1]) * activation_deriv(zetas[i-1]))
    
        #print("deltas[-1]:", deltas[-1].shape, "<==use ) a[", (i-2), "]:", 
        #      a[i-2].shape, " (next is wrong shape a[", (i-1),"]:", a[i-1].shape)
        #dW1 = 1/m * (dZ1.dot(X.T)) #10, 784
    
        #print("Next gradw:", deltas[-1].dot(a[i-2].T).shape )
        gradw.append( mm *deltas[-1].dot(acts[i-2].T)); # mm is correct
        # this is the wrong shape should use i-2
        #gradw.append(np.dot(a[i - 1].T, deltas[-1]))
    
        #gradb.append(np.sum(deltas[-1], axis=0, keepdims=True)) # this is good
        #print("deltas to work out gradb: m:", deltas[-1].shape[1])
        gradb.append( mm * np.sum(deltas[-1], 1)) # mm is correct - 1/mX sum(deltas)
        
    gradw.reverse()
    gradb.reverse()
    deltas.reverse()
    #print("***** FINISHED BACK PROP ***********")
    
    return deltas, gradw, gradb

def update_params(weights, biases, gradw, gradb, alpha):
    
    #print("******** UPDATE weights and biases *******")
    """
    for i in range(len(weights)):
        #print("w[", i, "]", w[i].shape, "gradw:", gradw[i].shape)
        weights[i] -= alpha*gradw[i]
        #print("b[", i, "]", b[i].shape, "gradb:", gradb[i].shape, "gradb:", gradb[i] )
        #b2 -= alpha * np.reshape(db2, (10,1))
        biases[i] -= alpha*gradb[i]
    #print("******** FINISH Update weight biases*****")
    """
    ### same code as above
    #np.reshape(db1, (10,1))
    for w, b, dW, dB in zip(weights, biases, gradw, gradb):
        w -= alpha*dW
        b -= alpha* np.reshape( dB, (b.shape[0], 1))
        
    return weights, biases
    
  
def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y)/Y.size
  
def gradient_descent(cfg, X, Y, alpha, iterations):
    weights, biases = init_params(cfg)
    
    for i in range(iterations):
        zetas, acts, A2 = forwardfeed(X, weights, biases)
        _, gradw, gradb = backward_propagate(weights, biases, zetas, acts, Y)
        weights, biases = update_params(weights, biases, gradw, gradb,  alpha)
        
        if (i+1) % (iterations/10) == 0:
        #if i % 10 == 0:
            print(f"Iteration: {i+1} / {iterations}")
            prediction = get_predictions(A2)
            print(f'{get_accuracy(prediction, Y):.3%}')
    
    return weights, biases

def make_predictions(X, weights, biases):
    _,_, A = forwardfeed(X, weights, biases)
    #_, _, _, A2 = forward_propagation(X, W1, b1, W2, b2)
    predictions = get_predictions(A)
    return predictions

def show_prediction(index,X, Y, weights, biases, img_width=28, img_height=28, img_scale_factor=255):
    vect_X = X[:, index,None]
    prediction = make_predictions(vect_X, weights, biases)
    label = Y[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = vect_X.reshape((img_width, img_height)) * img_scale_factor

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

##########MAIN ###########

cfg = [784, 20 , 15, 10]
alpha = 0.15


#X_train, Y_train, X_test, Y_test , WIDTH, HEIGHT, SCALE_FACTOR = read_data_mnist()
X_train, Y_train, X_test, Y_test, WIDTH, HEIGHT, SCALE_FACTOR = read_data_from_zip()

#w, b = init_params(cfg, n_layer)
#z,a, _ = forwardfeed(X, w, b, n_layer)
#deltas, gradw, gradb = backward_propagate(w, b, z, a, n_layer, Y)
#w, b = update_params(w, b, gradw, gradb, n_layer, alpha)

print("Start gradient descent")
weights, biases = gradient_descent(cfg, X_train, Y_train, alpha, 300)
print("gradient descent END")

""""
with open("trained_params.pkl","wb") as dump_file:
    pickle.dump((weights, biases),dump_file)

with open("trained_params.pkl","rb") as dump_file:
    weights, biases = pickle.load(dump_file)
"""

show_prediction(0,X_test, Y_test, weights, biases, WIDTH, HEIGHT, SCALE_FACTOR)
show_prediction(1,X_test, Y_test, weights, biases, WIDTH, HEIGHT, SCALE_FACTOR)
show_prediction(2,X_test, Y_test, weights, biases, WIDTH, HEIGHT, SCALE_FACTOR)
show_prediction(100,X_test, Y_test, weights, biases, WIDTH, HEIGHT, SCALE_FACTOR)
show_prediction(200,X_test, Y_test, weights, biases, WIDTH, HEIGHT, SCALE_FACTOR)




