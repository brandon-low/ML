import numpy as np
import pandas as pd
import json
import pickle
from keras.datasets import mnist
import matplotlib.pyplot as plt

debug = False

def generate_data(n: int) -> np.ndarray:
    return generate_sinwave_data(2, n)

def generate_sinwave_data(n: int, s: int) -> np.ndarray:
    x = np.linspace(0, 1, s)
    x = x.reshape(len(x), 1)
    y = np.sin(n * np.pi * x)
    return x, y

def generate_tanhwave_data(n: int) -> np.ndarray:
    x = np.linspace(-np.pi, np.pi, n)
    x = x.reshape(len(x), 1)
    y = np.tanh( x)
    return x, y

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


# tanh activation function
def tanh(x: np.ndarray) -> np.ndarray:
    # return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    return np.tanh(x)

# Derivative of Tanh Activation Function
def tanh_derv(x: np.ndarray) -> np.ndarray:
    return 1 - np.power(tanh(x), 2)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Sigmoid derivative for backpropagation
def derivative_sigmoid(x):
    #activation(x)*(1 - activation(x))
    return x*(1-x)

# Leaky_ReLU activation function
def leakyrelu(z, alpha):
    return np.max(alpha*z, z)
    
# Derivative of leaky_ReLU Activation Function
def dericative_leakyrelu(z, alpha):
    return 1 if z > 0 else alpha   

def ReLU(Z: np.ndarray) -> np.ndarray:
    # return max(0, z)
    return np.maximum(Z,0)

def derivative_ReLU(Z):
    #return 1 if z > 0 else 0
    return Z > 0

#Use for last layer
def softmax(Z: np.ndarray) -> np.ndarray:
    """Compute softmax values for each sets of scores in x."""
    exp = np.exp(Z - np.max(Z)) #le np.max(Z) evite un overflow en diminuant le contenu de exp
    return exp / exp.sum(axis=0)


def activation(x: np.ndarray) -> np.ndarray:
    return ReLU(x)
  
def activation_deriv(x: np.ndarray) -> np.ndarray:
    return derivative_ReLU(x)

def one_hot(Y: np.ndarray) -> np.ndarray:
    ''' return an 0 vector with 1 only in the position correspondind to the value in Y'''
    #print("One Hot Y:", Y.shape, "y.max():", Y.max(), "y.Size:", Y.size)
    #print("one hot Y:", Y)
    #one_hot_Y = np.zeros((Y.max()+1,Y.size)) #si le chiffre le plus grand dans Y est 9 ca fait 10 lignes
    one_hot_Y = np.zeros((Y.max()+1,Y.size))
    #print("one hot hotY:", one_hot_Y, " transposed:", one_hot_Y.T, " arrangeY:", np.arange(Y.size))
        
    one_hot_Y[Y,np.arange(Y.size)] = 1 # met un 1 en ligne Y[i] et en colonne i, change l'ordre mais pas le nombre
    #print("one hot hotY:", one_hot_Y, " transposed:", one_hot_Y.T, " arrangeY:", np.arange(Y.size))
      
    return one_hot_Y

#### Loading a Network
def load(filename: str):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
   
    weights = [np.array(w) for w in data["weights"]]
    biases = [np.array(b) for b in data["biases"]]
    return weights, biases


class NeuralNetwork:
    def __init__(self, cfg: []) -> None:
        
        self.cfg = cfg
        self.w = []
        self.b = []
        self.gradw = []
        self.gradb = []
        self.deltas = []
        np.random.seed(1)
        for i in range(len(cfg) -1):
            #print(nn_cfg[i+1] ,":", nn_cfg[i])
            #w.append(np.random.randn( cfg[i+1], cfg[i])) # dont use this 
         
            ## this is still the best
            self.w.append(np.random.normal(size=(cfg[i+1], cfg[i])) * np.sqrt(1./(cfg[i]*2)) )
            self.b.append(np.zeros((cfg[i+1], 1)) )
        
            if (i == 0):
                print("W:", self.w[i].shape,"B:", self.b[i].shape, "Input Layer")
            elif (i == (len(cfg)-2) ):
                print("W:", self.w[i].shape, "B:", self.b[i].shape, "Output Layer")
            else:
                print("W:", self.w[i].shape, "B:", self.b[i].shape, "Hidden Layer")
            
    def forwardTanh(self, x: np.ndarray) -> np.ndarray:
        self.z = []
        self.a = []
        self.z.append(x)
        self.a.append(x)
        
        if (debug): print("z[0]:", self.z[-1].shape, "a[0]:", self.a[-1].shape)
      
        num_layers =len(self.w)
    
        for i in range(num_layers):
            self.z.append(np.dot(self.w[i], self.a[i]) + self.b[i])
            #print("z[-1]:", z[-1].shape , "i:", i, "z[",(i+1),"]", z[i+1].shape, "z[",i,"]", z[i].shape)
            self.a.append(tanh(self.z[-1]))
            
            if (debug): print("z[", i, "]:", self.z[-1].shape, "a[", i, "]:", self.a[-1].shape)
        
        return self.a[-1]
     
    def backwardTanh(self, x: np.ndarray, y: np.ndarray) -> None:
        self.gradw = []
        self.gradb = []
        self.deltas = []
        
        self.deltas.append( (self.a[-1] - y.T))
        self.gradw.append( np.dot( self.deltas[-1], self.a[-2].T))
        self.gradb.append( np.sum(self.deltas[-1], 1))
         
        num_layers = len(self.w)
        for i in range(num_layers, 1, -1):     # loop is right 
            #print("[", i, "] w:", self.w[i-1].T.shape , "deltas[-1]:", self.deltas[-1].shape, )
            self.deltas.append(np.dot(self.w[i-1].T, self.deltas[-1]) * tanh_derv(self.z[i-1]))
            self.gradw.append( np.dot( self.deltas[-1], self.a[i-2].T))
            self.gradb.append( np.sum(self.deltas[-1], 1))
            #print("[", i, "] deltas:", self.deltas[-1].shape, "gradw:", self.gradw[-1].shape, "gradb:", self.gradb[-1].shape, "=", self.gradb[-1])
            #if (debug): print("deltas[", i,"]:", self.deltas[-1].shape  , "gradw[", i, "]:", self.gradw[-1].shape, "gradb[", i, "]:", self.gradb[-1].shape)
       
        self.gradw.reverse()
        self.gradb.reverse()
        self.deltas.reverse()
        if (debug):
            for dd, dW, dB in zip(self.deltas, self.gradw, self.gradb):
                print("Deltas:", dd.shape  , "gradw:", dW.shape, "gradb:", dB.shape)
    
    def updateTanh(self, lr: float) -> None:
        for i in range( len(self.w) ):
            self.w[i] -= lr*self.gradw[i]
            #self.b[i] -= lr*self.gradb[i]
            self.b[i] -= lr*np.reshape(self.gradb[i],(self.b[i].shape[0], 1)) 
            #lr* np.reshape( dB, (b.shape[0], 1))
            #self.b[i] -= lr* np.reshape( self.gradb[i], (self.b[i].shape[0], 1))
     
    def predictTanh(self, x: np.ndarray) -> np.ndarray:
        return self.forwardTanh(x)

    def mse(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.mean(np.power(self.predictTanh(x) - y, 2))
    
    def train(self, x: np.ndarray, y: np.ndarray, lr: float=0.001, epochs: int=1000) -> None:
        for i in range(epochs):
            A2 = self.forwardTanh(x)
            #print("A2:", A2)
            self.backwardTanh(x, y)
            self.updateTanh(lr)
            # if (self.mse(x, y) < 0.1):
                # break
            if i % 100 == 0:
                print(f'Epoch {i}: {self.mse(x, y):.3%}')
            
    def setWeightsBiases(self, w: [], b: [])-> None:
        self.w = w
        self.b = b
           
    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {
                "cfg:": [self.cfg],
                "weights": [w.tolist() for w in self.w],
                "biases": [b.tolist() for b in self.b]}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.z = []
        self.a = []
        self.z.append(x)
        self.a.append(x)
        num_layers = len(self.w)
    
        #print("**** FORWARD FEED  return the last activation which is the result ******")
        for i in range(num_layers):
            self.z.append( np.dot(self.w[i], self.a[i]) + self.b[i])
            #print("z[-1]:", z[-1].shape , "i:", i, "z[",(i+1),"]", z[i+1].shape, "z[",i,"]", z[i].shape)
        
            if (i == (num_layers-1) ):
                #print("Forward Output Layer W:", w[i].shape)
                self.a.append(softmax(self.z[i+1]))
            else:
                self.a.append(ReLU(self.z[i+1]))
        
        #print("ForwardFeed i:", i, "z[-1]:", z[-1].shape, "a[-1]:", a[-1].shape)
        #print("return at forward a[-1]:", a[-1].shape)
        #print("**** FORWARD FEED FINISH******")
        return self.a[-1]
        
    def backward(self, x: np.ndarray, y: np.ndarray) -> None:
        self.gradw = []
        self.gradb = []
        self.deltas = []
        
        one_hot_Y = one_hot(y)
        self.deltas.append( 2 * (self.a[-1] - one_hot_Y))    #dZ2 = 2*(A2 - one_hot_Y) #10,m
        one_div_m = 1/self.deltas[-1].shape[1]      ## same number for all
        self.gradw.append(one_div_m * np.dot(self.deltas[-1], self.a[-2].T) ) # delta[out neurons, m]
        self.gradb.append(one_div_m * np.sum(self.deltas[-1], 1)) # 1/mX sum(deltas)
        
        num_layers = len(self.w)
        for i in range(num_layers, 1, -1):     # loop is right 
            self.deltas.append( np.dot(self.w[i-1].T, self.deltas[-1]) * derivative_ReLU(self.z[i-1]))
            self.gradw.append( one_div_m * np.dot(self.deltas[-1], self.a[i-2].T)); # mm is correct
            self.gradb.append( one_div_m * np.sum(self.deltas[-1], 1)) # mm is correct - 1/mX sum(deltas)
     
        
        self.gradw.reverse()
        self.gradb.reverse()
        self.deltas.reverse()
        
    def update(self, lr: float) -> None:
        for w, b, dW, dB in zip(self.w, self.b, self.gradw, self.gradb):
            w -= lr*dW
            b -= lr* np.reshape( dB, (b.shape[0], 1))
            
    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
  
    def get_predictions(self, y_pred: np.ndarray) -> np.ndarray:
        #print("A2:", A2.shape);
        return np.argmax(y_pred, 0)

    def get_accuracy(self, predictions: np.ndarray, Y: np.ndarray):
        #print("pred:", predictions.shape, "y:", Y.shape)
        return np.sum(predictions == Y)/Y.size
  
    def gradient_descent(self, x: np.ndarray, y: np.ndarray, lr: float=0.15, epochs: int=200) -> None:    
        for i in range(epochs):
            y_pred = self.forward(x)
            self.backward(x, y)
            self.update(lr)
               
            if (i+1) % (epochs/10) == 0:
            #if i % 10 == 0:
                print(f"Iteration: {i+1} / {epochs}")
                prediction = self.get_predictions(y_pred)
                print(f'{self.get_accuracy(prediction, y):.3%}')
        

    def make_predictions(self, x: np.ndarray) -> np.ndarray:
        y_pred = self.forward(x)
        predictions = self.get_predictions(y_pred)
        return predictions

    def show_prediction(self, index: int , X: np.ndarray, Y:  np.ndarray, img_width: int=28, img_height: int=28, img_scale_factor: int=255)-> None:    
        vect_X = X[:, index,None]
        prediction = self.make_predictions(vect_X)
        label = Y[index]
        print("Prediction: ", prediction)
        print("Label: ", label)

        current_image = vect_X.reshape((img_width, img_height)) * img_scale_factor

        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.show()



##########MAIN ###########


x, y = generate_sinwave_data(2, 100)
#x, y = generate_tanhwave_data(100)

nn = NeuralNetwork([1, 50, 50, 40, 1])
x=x.T
print("Start Training: x:", x.shape, "y:", y.shape )
nn.train(x, y,  0.0011, 2000)
print("End Training!")

y_pred = nn.predictTanh(x)


#plt.plot(x[0], y, color ='tab:pink')
plt.scatter(x, y, c="pink",linewidths = 0.3)
#plt.scatter(x, y_pred, c ='blue') 
plt.plot(x[0], y_pred[0], color ='tab:blue') 

plt.show()

filename = "test_graph.json"
nn.save(filename)

w, b= load(filename)



"""


cfg = [784, 10, 10, 10]
alpha = 0.15


#X_train, Y_train, X_test, Y_test , WIDTH, HEIGHT, SCALE_FACTOR = read_data_mnist()
X_train, Y_train, X_test, Y_test, WIDTH, HEIGHT, SCALE_FACTOR = read_data_from_zip()

#w, b = init_params(cfg, n_layer)
#z,a, _ = forwardfeed(X, w, b, n_layer)
#deltas, gradw, gradb = backward_propagate(w, b, z, a, n_layer, Y)
#w, b = update_params(w, b, gradw, gradb, n_layer, alpha)
neuralNetwork = NeuralNetwork(cfg)

print("Start gradient descent: x:", X_train.shape, "y:", Y_train.shape )
neuralNetwork.gradient_descent( X_train, Y_train, alpha, 200)
print("gradient descent END")


#with open("trained_params.pkl","wb") as dump_file:
#    pickle.dump((weights, biases),dump_file)

#with open("trained_params.pkl","rb") as dump_file:
#    weights, biases = pickle.load(dump_file)


neuralNetwork.show_prediction(0,X_test, Y_test, WIDTH, HEIGHT, SCALE_FACTOR)
neuralNetwork.show_prediction(1,X_test, Y_test, WIDTH, HEIGHT, SCALE_FACTOR)
neuralNetwork.show_prediction(2,X_test, Y_test, WIDTH, HEIGHT, SCALE_FACTOR)
neuralNetwork.show_prediction(100,X_test, Y_test, WIDTH, HEIGHT, SCALE_FACTOR)
neuralNetwork.show_prediction(200,X_test, Y_test, WIDTH, HEIGHT, SCALE_FACTOR)

"""


