
import numpy as np
import json

def init_params(cfg):
    weights = []
    biases = []
    np.random.seed(1)
    print("init_params")
    for i in range(len(cfg) -1):
        #print(cfg[i+1] ,":", cfg[i])
       
        ## this is still the best
        weights.append(np.random.normal(size=(cfg[i+1], cfg[i])) * np.sqrt(1./(cfg[i]*2)) )
        biases.append(np.zeros((cfg[i+1], 1)))
        
        if (i == 0):
            print("W:", weights[i].shape,"B:",biases[i].shape, "Input Layer")
        elif (i == (len(cfg)-2) ):
            print("W:", weights[i].shape, "B:",biases[i].shape, "Output Layer")
        else:
            print("W:", weights[i].shape, "B:",biases[i].shape, "Hidden Layer")
        
    return weights, biases

def dump(weights: [], biases:[], config:[]):
    print("cfg:", cfg)
    for w, b in zip(weights, biases):
        print("W:", w.shape,"B:", b.shape)
#for b, w in zip(self.biases, self.weights):

cfg = [784, 50, 30, 10]
#n_layer = len(cfg)

weights, biases = init_params(cfg)

print("Inited W, B, len(w):", len(weights), "len(cfg):", len(cfg))
dump(weights, biases, cfg)
    
print("cfg[-1]:", cfg[-1], "cfg[-2]:", cfg[-2])
    
fname ="test.json"
data = {
    "config": cfg,
    "weights": [w.tolist() for w in weights],
    "biases": [b.tolist() for b in biases]}
f = open(fname, "w")
json.dump(data, f)
f.close()

f = open(fname, "r")
ldata = json.load(f)
f.close()
   
lcfg = ldata["config"]
w1 = [np.array(w) for w in ldata["weights"]]
b1 = [np.array(b) for b in ldata["biases"]]

print("Loaded W, B:")
dump(w1, b1, lcfg)
    
    
    