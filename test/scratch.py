#import zipfile
import os
import pandas as pd
import numpy as np


def ReLU(Z):
    return np.maximum(Z,0)

def derivative_ReLU(Z):
    return Z > 0

def softmax(Z):
    """Compute softmax values for each sets of scores in x."""
    exp = np.exp(Z - np.max(Z)) #le np.max(Z) evite un overflow en diminuant le contenu de exp
    return exp / exp.sum(axis=0)


#### Activation functions

# Sigmoid derivative for backpropagation
def sigmoid_deriv(x):
        return x*(1-x)
    
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def feedforward(weights, biases, inputs):
        """Return the output of the network if ``inputs`` is input."""
        for b, w in zip(biases, weights):
            inputs = np.dot(inputs, w.T)
            #inputs = sigmoid(np.dot(inputs, w.T)+b)
        return inputs

def feedforward2(weights, biases, inputs):
        """Return the output of the network if ``inputs`` is input."""
        for b, w in zip(biases, weights):
            inputs = np.dot(inputs, w.T)+b
            #inputs = sigmoid(np.dot(inputs, w.T)+b)
        return inputs
    
def dumpNpArray(title, v):
    print(title, ":" , v)
    print(title, " Size:", len(v))

    for i in range(len(v)) :
        print(title, "[",i, "] size:", v[i].size, " val:", v[i])

def getNpArray(z, n=0):
    print("getNpArray:", len(z), "n:", n)
    if n < len(z):
        return z[n]
    else:
        return []
    
def isLast(z, n):
    return (n == len(z) -1)      
    
def setAll(v, x = 1):
    print("In set All")
    for i in range(len(v)) :
        row = v[i]
        print("row len:", len(row))
        for j in range(len(row)):
            e = row[j]
            print("element:", e, " len:", len(e))
            for n in range(len(e)):
                print("k[", n,"]:", e[n])
                e[n] = 1
    
def testDump(weights, biases):
    dumpNpArray("Biases", biases)
    dumpNpArray("Weights", weights)

    print("Layer 0,                             Weight[0]:", weights[0], "len:", len(weights[0]))
    print("Layer 0,             Nueron Weight 0, Weight[0]:", weights[0][0], "len:",len(weights[0][0]) )
    print("Layer 0, Nueron Weight 0, Element [0] Weight[0]:", weights[0][0][0])

    print("Layer 0", getNpArray(weights, 0))
    print("Layer 1", getNpArray(weights, 1))
    print("Layer 2", getNpArray(weights, 2))
    print("Is Last:", isLast(weights, 0) ," should be false")
    print("Is Last:", isLast(weights, 1) ," should be true")
    print("Is Last:", isLast(weights, 2) ," should be false")
    
def isEmpty(x, i):
    if len(x) == 0:
        return True
    return len(x) <= i

################ MAIN ######################
# The training set, with 4 examples consisting of 3 input values and 1 output value
training_inputs = np.array([    [0, 0, 1],
                                [1, 1, 1],
                                [1, 0, 1],
                                [0, 1, 1] ]).T

#transpose it to trun into 4X1 matrix
training_outputs = np.array([[0,1,1,0]]).T

# Neuron List of 3 Layers. 15 input, 10 hidden, 5 out put
neuron_list = [3, 2, 1]
layers = len(neuron_list)
print("NeuronList:", neuron_list, " layers:", layers )

## Init biases
np.random.seed(1)
#biases = [np.random.randn(y, 1) for y in neuron_list[1:]]

## performance init
#biases = [np.random.normal(size=(y, 1)) * np.sqrt(1./y) for y in neuron_list[1:] ]
biases = [np.zeros(y)+1 for y in neuron_list[1:]]

  
# ORG INIT
#weights = [np.random.randn(y, x)
#                        for x, y in zip(neuron_list[:-1], neuron_list[1:])]
## set to test
weights = [np.random.randn(y, x)
                        for x, y in zip(neuron_list[:-1], neuron_list[1:])]
#for w in weights[1:]]
#    w = 0
    
#performance
#weights = [np.random.normal(size=(y, x)) * np.sqrt(1./(x))
#                        for x, y in zip(neuron_list[:-1], neuron_list[1:])]
# W1 = np.random.normal(size=(10, size)) * np.sqrt(1./(size))
#    W2 = np.random.normal(size=(10, 10)) * np.sqrt(1./20)

#setAll(weights)

#dumpNpArray("Weights", weights)
#testDump(weights, biases)
print("Training Inputs:" )
print( training_inputs)
print("Training Outputs:" )
print( training_outputs)

outputs = []



for i in range(layers-1):
    num_of_synaptic_weights = neuron_list[i]
    #print("Layer[", i,"] has", num_of_synaptic_weights, " synaptic_weights")
    #print("Weight:", weights[i], "lengthOFWeight:", len(weights[i]), "Means that many resultants nuerons to be activated")
    #print("Biases:", biases[i], "lengthOFBiases:", len(biases[i]), "Means that many resultants nuerons to be activated")

    weights_at_layer = weights[i];
    biases_at_layer = biases[i];
    #print("IS Last:", isLast(weights, i))
    print("W:", weights_at_layer)
    print("B:", biases_at_layer)
  
    for j in range(len(weights_at_layer)):
        # print("Loop: [i,j]:[",i,",", j ,"]", end=" ")
        input_layer = training_inputs
        w = weights_at_layer[j] # array of weights
        b = biases_at_layer[0]  # single value
        input_layer = training_inputs
        if i==0 :
            print("at Layer 0 use training input")
            input_layer = training_inputs
        else:
            print("at layer:",i ,"use results stored from previous activation")
            print("output len:", len(outputs))
            input_layer = outputs[i-1][j]
            
        print("[", j, "] w:", w, " b:", b)
        
        lout = np.dot(input_layer, w)
        
        print("lout:", lout)
        if isEmpty(outputs, i):
            #1st element just append it
            outputs.append([lout])
        else:
            outputs[i].append(lout)
            
        print(outputs)
        # here is where we forwrd feed
         
   
    
    
    
#outputs = feedforward(weights, biases, input_layer)
#print("Output:")
#print(outputs)

#print("Output2:", feedforward2(weights, biases, input_layer))



 ##Return the output of the network if ``a`` is input.
#for i in range(layers-1):

 #   print("Loop [", i, "] Weights[",i,"]:", weights[i])
#    outputs = feedforward(weights[i], biases[i], input_layer)
#    print("outputs:", outputs)
   

"""
fname = "../prototype/train.csv.zip"
if os.path.exists(fname):
        print("about to open file")
        
        # read the dataset using the compression zip
        df = pd.read_csv(fname,compression='zip')
 
        # display dataset
        print(df.head())
        
        data = pd.read_csv("../prototype/train.csv")
        
        print(data.head())
        
        #print ("compare:", df.compare(data, keep_shape=True, keep_equal=True))
else:
        print("file do not exist")
"""