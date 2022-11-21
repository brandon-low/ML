
import numpy as np
import pandas as pd


def init_params(cfg):
    weights = []
    biases = []
    np.random.seed(1)
    print("init_params")
    for i in range(len(cfg) -1):
        #print(cfg[i+1] ,":", cfg[i])
       
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

#for b, w in zip(self.biases, self.weights):

cfg = [784, 50, 30, 10]
#n_layer = len(cfg)

weights, biases = init_params(cfg)

print("len(w):", len(weights), "len(cfg):", len(cfg))
for w, b in zip(weights, biases):
     print("W:", w.shape,"B:", b.shape)

# last is len(weights) -1
for i in range(len(weights)):
    print("i:", i, "W:", w.shape,"B:", b.shape)
    
print("cfg[-1]:", cfg[-1], "cfg[-2]:", cfg[-2])
    
 
    
    
    