import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def activation(x: np.ndarray) -> np.ndarray:
    # x = [-1*num for num in x]
    # return (1)/(1 + np.exp(x))
    return np.tanh(x)

def activation_deriv(x: np.ndarray) -> np.ndarray:
    # return activation(x)*(1 - activation(x))
    return 1 - np.power(activation(x), 2)
  
def forward(x: np.ndarray, n_layer: int):
    z = []
    a = []
    z.append(x)
    a.append(x)
    print("**** FORWARD FEED  return the last activation which is the result ******")
    for i in range(n_layer - 1):
        z.append(w[i].dot(a[i]) + b[i])
        #print("z[-1]:", z[-1].shape , "i:", i, "z[",(i+1),"]", z[i+1].shape, "z[",i,"]", z[i].shape)
        a.append(activation(z[i+1]))
        print("ForwardFeed i:", i, "z[-1]:", z[-1].shape, "a[-1]:", a[-1].shape)
    
    print("return at forward a[-1]:", a[-1].shape)
    return z, a, a[-1]

     
print("Tear Down NN")

"""
   n_input: dimensionality of the input
   n_hidden: number of neurons in the hidden layers
   n_output: dimensionality of the output
   num_layers: the number of hidden layers
"""

nn_cfg = [784, 100, 30, 20, 10]

nn_cfg_n_layers = len(nn_cfg)
w = []
b = []
gradw = []
gradb = []
deltas = []

print("cfg_layers:", nn_cfg_n_layers)
for i in range(nn_cfg_n_layers -1):
    #print(nn_cfg[i+1] ,":", nn_cfg[i])
    w.append(np.random.randn( nn_cfg[i+1], nn_cfg[i]))
    if (i == 0):
        print("Input Layer W:", w[i].shape)
    elif (i == (nn_cfg_n_layers-2) ):
        print("Output Layer W:", w[i].shape)
    else:
        print("Hidden Layer:W", w[i].shape)
  

for i in range(nn_cfg_n_layers -1):
    b.append(np.zeros((nn_cfg[i+1], 1)))
    if (i == 0):
        print("Input Bias:", b[i].shape )
    elif (i == (nn_cfg_n_layers-2) ):
        print("Output Bias:", b[i].shape )
    else:
        print("Hidden Bias:", b[i].shape )



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

#np.random.shuffle(data) 

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / SCALE_FACTOR

print("X_Train shape:", X_train.shape )
print("Y_Train shape:", Y_train.shape , " shape[0]:", Y_train.shape[0])

x = X_train

z = []
a = []
z.append(X_train)
a.append(X_train)
print("**** FORWARD FEED  return the last activation which is the result ******")
for i in range(nn_cfg_n_layers -1):
    z.append(w[i].dot(a[i]) + b[i])
    #print("z[-1]:", z[-1].shape , "i:", i, "z[",(i+1),"]", z[i+1].shape, "z[",i,"]", z[i].shape)
    a.append(activation(z[i+1]))
    print("ForwardFeed i:", i, "z[-1]:", z[-1].shape, "a[-1]:", a[-1].shape)
    
print("return at forward a[-1]:", a[-1].shape)

print("**** BACK PROP  START ****")
gradw = []
gradb = []
deltas = []

#print("delta:", a[-1].shape, "-", Y_train.shape , "=", (a[-1] - Y_train).shape )

deltas.append(a[-1] - Y_train)  # correct

#print("a[-2]:", a[-2].shape, " a[-2].T:", a[-2].T.shape, "deltas[-1]:", deltas[-1].shape)
print("DoLast gradw: " , deltas[-1].dot(a[-2].T).shape)

total_m = deltas[-1].shape[1]
print("deltas to work out gradw & gradb: m:", total_m)
mm = 1/total_m
#gradw.append(np.dot(a[-2].T, deltas[-1])) # wrong size
gradw.append(mm* deltas[-1].dot(a[-2].T) ) # delta[out neurons, m]
gradb.append(mm * np.sum(deltas[-1])) # 1/mX sum(deltas)

#gradb.append(np.sum(deltas[-1], axis=0, keepdims=True))

print("Start Loop Range deltas:" , len(deltas), " w:", len(w), "z:", len(z))
print("gradw:", gradw[-1].shape, "gradb:", gradb[-1].shape)
print("Size a:", len(a) , "a[-1]", a[-1].shape, "a[", (len(a)-1), "]:", a[len(a)-1].shape,
        "a[",  (len(a)-2), "]:", a[len(a)-2].shape)

print("***** BUILD Rest Of Deltas , gradw, gradb ******")
#for i in range(nn_cfg_n_layers - 1, 0, -1):     # loop is wrong should stop at 1
for i in range(nn_cfg_n_layers - 1, 1, -1):     # loop is right 
    print("DO:", i);
    ## should use w [i-1]
    print("z[",(i-1), "]", z[i-1].shape, "w[",i,"]", w[i-1].shape)
    print("activate_dev:", activation_deriv(z[i]).shape)
    print("w[i-1].T:", w[i-1].T.shape)
    print("deltas[-1]:", deltas[-1].shape)
    print("dZ=", w[i-1].T.dot(deltas[-1]).shape)
    #dZ1 = W2.T.dot(dZ2)*derivative_ReLU(Z1) # 10, m
    deltas.append( w[i-1].T.dot(deltas[-1]) * activation_deriv(z[i-1]))
    
    print("deltas[-1]:", deltas[-1].shape, "<==use ) a[", (i-2), "]:", a[i-2].shape, " (next is wrong shape a[", (i-1),"]:", a[i-1].shape)
    #dW1 = 1/m * (dZ1.dot(X.T)) #10, 784
    
    total_m = deltas[-1].shape[1]   
    print("deltas to work out gradw & gradb: m:", total_m)
    mm = 1/total_m
    print("Next gradw:", deltas[-1].dot(a[i-2].T).shape )
    gradw.append(mm*deltas[-1].dot(a[i-2].T));
    # this is the wrong shape should use i-2
    #gradw.append(np.dot(a[i - 1].T, deltas[-1]))
    
    #gradb.append(np.sum(deltas[-1], axis=0, keepdims=True)) # this is good
    print("deltas to work out gradb: m:", deltas[-1].shape[1])
    gradb.append( mm * np.sum(deltas[-1])) # 1/mX sum(deltas)

print("***** FINISHED BACK PROP ***********")
gradw.reverse()
gradb.reverse()
deltas.reverse()


print("Size gradw:", len(gradw), "gradw[0]:", gradw[0].shape)
print("Size gradb:", len(gradb), "gradb[0]:", gradb[0].shape)
print("Size deltas:", len(deltas), "deltas[0]:", deltas[0].shape)

print("******** UPDATE weights and biases *******")
alpha = 0.15
for i in range(nn_cfg_n_layers -1):
    print("w[", i, "]", w[i].shape, "gradw:", gradw[i].shape)
    w[i] -= alpha*gradw[i]
    print("b[", i, "]", b[i].shape, "gradb:", gradb[i].shape, "gradb:", gradb[i] )
    #b2 -= alpha * np.reshape(db2, (10,1))
    b[i] -= alpha*gradb[i]

print("******** FINISH Update weight biases*****")
    

