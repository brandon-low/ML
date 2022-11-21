import numpy as np
import matplotlib.pyplot as plt

"""
- [x] generate data
- [ ] neural network
    - [x] init
    - [ ] forward
    - [ ] backpropagate
    - [x] activation
    - [ ] train
"""

### GENERATE SIN WAVE ############
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

class NeuralNetwork:
    def __init__(self, cfg: []) -> None:
        self.w = []
        self.b = []
        self.gradw = []
        self.gradb = []
        self.deltas = []
        np.random.seed(1)
        for i in range(len(cfg) -1):
            ## this is still the best
            self.w.append(np.random.normal(size=(cfg[i], cfg[i+1])) * np.sqrt(1./(cfg[i+1])) )
            self.b.append(np.zeros((1, cfg[i+1])))
        
            if (i == 0):
                print("W:", self.w[i].shape,"B:", self.b[i].shape, "Input Layer")
            elif (i == (len(cfg)-2) ):
                print("W:", self.w[i].shape, "B:", self.b[i].shape, "Output Layer")
            else:
                print("W:", self.w[i].shape, "B:", self.b[i].shape, "Hidden Layer")
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Sigmoid derivative for backpropagation
    def sigmoid_deriv(self, x):
        #activation(x)*(1 - activation(x))
        return self.sigmoid(x) *(1 - self.sigmoid(x))

    def tanh(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    def tanh_deriv(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.power(self.tanh(x), 2)
    
    def activation(self, x: np.ndarray) -> np.ndarray:
        # x = [-1*num for num in x]
        # return (1)/(1 + np.exp(x))
        return self.tanh(x)
        #return self.sigmoid(x)

    def activation_deriv(self, x: np.ndarray) -> np.ndarray:
        # return self.activation(x)*(1 - self.activation(x))
        return self.tanh_deriv(x)
        #return self.sigmoid_deriv(x)
    
    def ReLU(self, z: np.ndarray) -> np.ndarray:
        # return max(0, z)
        return np.maximum(z,0)
    
    def derivative_ReLU(self, Z):
        #return 1 if z > 0 else 0
        return Z > 0
        
    def softmax(self, z: np.ndarray) -> np.ndarray:
        """Compute softmax values for each sets of scores in x."""
        exp = np.exp(z - np.max(z)) #le np.max(Z) evite un overflow en diminuant le contenu de exp
        return exp / exp.sum(axis=0)
    
    def one_hot(self, Y: np.ndarray) -> np.ndarray:
        ''' return an 0 vector with 1 only in the position correspondind to the value in Y'''
        one_hot_Y = np.zeros((Y.max()+1,Y.size)) #si le chiffre le plus grand dans Y est 9 ca fait 10 lignes
        one_hot_Y[Y,np.arange(Y.size)] = 1 # met un 1 en ligne Y[i] et en colonne i, change l'ordre mais pas le nombre
        return one_hot_Y

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def mse(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.mean(np.power(self.predict(x) - y, 2))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.z = []
        self.a = []
        self.z.append(x)
        self.a.append(x)
     
        nWeights= len(self.w)
    
        #print("**** FORWARD FEED  return the last activation which is the result ******")
        for i in range(nWeights):
            self.z.append(np.dot(self.a[i], self.w[i]) + self.b[i])
            #print("[",i,"] a[i]:", self.a[i].shape, "w[i]:", self.w[i].shape, 
            #      "dot:", (np.dot(self.a[i], self.w[i]) + self.b[i]).shape,
            #     "a[i+1]:", self.activation(self.z[i + 1]).shape )
            self.a.append(self.activation(self.z[i + 1]))
            
            """
            self.z.append(self.w[i].dot(self.a[i]) + self.b[i])
            #print("z[-1]:", z[-1].shape , "i:", i, "z[",(i+1),"]", z[i+1].shape, "z[",i,"]", z[i].shape)
        
            if (i == (nWeights-1) ):
                #print("Forward Output Layer W:", w[i].shape)
                self.a.append( self.softmax(self.z[i+1]) )
            else:
                self.a.append( self.ReLU(self.z[i+1]) )
            """
        #print("ForwardFeed i:", i, "z[-1]:", z[-1].shape, "a[-1]:", a[-1].shape)
    
        #print("return at forward a[-1]:", a[-1].shape)
        #print("**** FORWARD FEED FINISH******")
        return self.a[-1]

    def backward(self, x: np.ndarray, y: np.ndarray) -> None:
        self.gradw = []
        self.gradb = []
        self.deltas = []
        
        self.deltas.append(self.a[-1] - y)
        #print("a[-1]:", self.a[-1].shape, "y:", y.shape , "self.a[-1] - y:", (self.a[-1] - y).shape )
        self.gradw.append(np.dot(self.a[-2].T, self.deltas[-1]))
        #print("gradw [0]:", np.dot(self.a[-2].T, self.deltas[-1]).shape )
        self.gradb.append(np.sum(self.deltas[-1], axis=0, keepdims=True))
        
        
        """
        one_hot_Y = self.one_hot(y)
        self.deltas.append( 2 * (self.a[-1] - one_hot_Y))    #dZ2 = 2*(A2 - one_hot_Y) #10,m    
        total_m = self.deltas[-1].shape[1]
        mm = 1/total_m      ## same number for all
        self.gradw.append(mm * self.deltas[-1].dot(self.a[-2].T) ) # delta[out neurons, m]
        self.gradb.append(mm * np.sum(self.deltas[-1], 1)) # 1/mX sum(deltas)
        """


        nWeights=len(self.w)
        for i in range(nWeights, 1, -1):     # loop is right
            #print("z[", i,"]:", (np.dot(self.deltas[-1], self.w[i-1].T) * self.activation_deriv(self.z[i-1])).shape )
            self.deltas.append(np.dot(self.deltas[-1], self.w[i-1].T) * self.activation_deriv(self.z[i-1]))
            
            #print("gradw[", i, ":", (np.dot(self.a[i - 2].T, self.deltas[-1])).shape)
            self.gradw.append(np.dot(self.a[i - 2].T, self.deltas[-1]))
            self.gradb.append(np.sum(self.deltas[-1], axis=0, keepdims=True))
            
            """
            self.deltas.append( self.w[i-1].T.dot(self.deltas[-1]) * self.derivative_ReLU(self.z[i-1])) 
            self.gradw.append( mm * self.deltas[-1].dot(self.a[i-2].T)); # mm is correct
            self.gradb.append( mm * np.sum(self.deltas[-1], 1)) # mm is correct - 1/mX sum(deltas)
            """
        self.gradw.reverse()
        self.gradb.reverse()
        self.deltas.reverse()
    
    def update(self, lr: float) -> None:
        for i in range( len(self.w) ):
            self.w[i] -= lr*self.gradw[i]
            self.b[i] -= lr*self.gradb[i]
            #self.b[i] -= lr* np.reshape( self.gradb[i], (self.b[i].shape[0], 1))
        
    def gradient_descent(self, x: np.ndarray, y: np.ndarray, lr: float, epochs: int) -> None:
        for i in range(epochs):
            self.forward(x)
            self.backward(x, y)
            self.update(lr)
            #if (self.mse(x, y) < 0.1):
            #    break
            if i % 100 == 0:
                print(f'Epoch {i}: {1-self.mse(x, y):.3%}')


######   MAIN ############

#make an NeuralNetwork
x, y = generate_sinwave_data(2, 100)
#x, y = generate_tanhwave_data(200)
print("x:", x.shape, "y:", y.shape)

neuralNetwork = NeuralNetwork([1, 50, 50, 40, 1])

# original train is 10000
neuralNetwork.gradient_descent(x, y, 0.0001, 10000)


y_pred = [np.mean(a) for a in neuralNetwork.predict(x)]
plt.scatter(x, y, c="pink",linewidths = 0.3)
plt.plot(x, y_pred)
plt.show()

