import numpy as np




def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
def sigmoid_deriv(x):
        return x*(1-x)
    
def ReLU(Z):
    return np.maximum(Z,0)

def derivative_ReLU(Z):
    return Z > 0

def softmax(Z):
    """Compute softmax values for each sets of scores in x."""
    exp = np.exp(Z - np.max(Z)) #le np.max(Z) evite un overflow en diminuant le contenu de exp
    return exp / exp.sum(axis=0)

def one_hot(Y):
    ''' return an 0 vector with 1 only in the position correspondind to the value in Y'''
    one_hot_Y = np.zeros((Y.max()+1,Y.size)) #si le chiffre le plus grand dans Y est 9 ca fait 10 lignes
    one_hot_Y[Y,np.arange(Y.size)] = 1 # met un 1 en ligne Y[i] et en colonne i, change l'ordre mais pas le nombre
    return one_hot_Y

def Activate(Z):
    return ReLU(Z)
    #return sigmoid(Z)

def dumpNpArray(title, v):
    print(title, ":" , v)
    print(title, " Size:", len(v))

    for i in range(len(v)) :
        print(title, "[",i, "] size:", v[i].size, " val:", v[i])

def init_params(neuron_list):
    ## Init biases
    np.random.seed(1)

    
    weights=[]
    biases = []
    for i in range(len(neuron_list)-1):
        weights.append(np.random.randn(neuron_list[i+1], neuron_list[i]))
        biases.append( np.zeros( ( neuron_list[i+1],1) ) )
    
    return weights, biases

def forward_feed(weights, biases, t):
    lzetas=[]
    lacts=[]
    for i in range(len(weights)):
        if i == 0:
            t_in = t
        else:
            t_in = lacts[i-1]
        
        w = weights[i]
        b = biases[i]
        z = w.dot(t_in)+b
    
        # doing last
        if (i == len(weights) -1) :
            a = softmax(z)
        else:
            a = ReLU(z)
            
        #a = sigmoid(z)
        ## save it
        lzetas.append(z)
        lacts.append(a)
    
        print("F[",i, "] w:", w.shape, "t:", t_in.shape, "z:",z.shape, " a:", a.shape )
    
    return lzetas, lacts
   
def back_prop(weights, biases, zetas, acts, alpha, t, t_expected):
    num = len(zetas)
    size , m = t.shape
    print ("****Back Prop**** zetas:", num, " m:" , m)
    for i in reversed(range(num )) :
        print("do:", i, " w:", len(weights) ," isLast:", (i == len(weights)-1))
        weight = weights[i]
        bias = biases[i]
        a_out = acts[i]
        z_out = zetas[i]
        if (i >= 1):
            a_in = acts[i-1]
        else:
            a_in = t
            
        print("w(current):" , weight.shape, " a:", a_in.shape ,"==> result a:", a_out.shape, ":", z_out.shape)
        if ((i == len(weights)-1)):
            print("doLast")
            one_hot_Y = t_expected
            print("a_out:", a_out.shape, " hotY:", one_hot_Y.shape)
            dZ2 = 2*(a_out - one_hot_Y) #10,m
            dW2 = 1/m * (dZ2.dot(a_in.T)) # 10 , 10
            db2 = 1/m * np.sum(dZ2,1) # 10, 1 return (1,)
            
            bSize, bM = bias.shape
            print("Shape Bias:", bSize,":", bM  , " db2:", db2)
            bias = bias - ( alpha * np.reshape(db2, (bM,1)) )
            print("dZ:", dZ2.shape, "dw:", dW2.shape, "should be same as w:", weight.shape, "db:", db2.shape)
            print("dw:", dW2)
            print("alpha*dw:", alpha*dW2)
            print("+")
            print("w:", weight)
            print("-=")
            weight -= alpha * dW2
            print("new Weight:", weight)
        else:
            print("do Others")
            w2 = weights[i+1]
            z1 = z_out
            print("derv(z1):", derivative_ReLU(z1).shape) 
            dZ1 = w2.T.dot(dZ2)*derivative_ReLU(z1) # 10, m
            print("dZ1", dZ1.shape)
            dW1 = 1/m * (dZ1.dot(a_in.T)) #10, 784
             
            bSize, bM = bias.shape
            print("Shape Bias:", bSize,":", bM  )
            db1 = 1/m * np.sum(dZ1, bM) # 10, 1
             
            print("db:" ,db1, ":", db1.shape, " bias=" , bias.shape)
            print("aaphpa*db1=", alpha * db1)
            bias = bias - db1
            print("dZ:", dZ1.shape, "dw:", dW1.shape, "should be same as w:", weight.shape, "db:", db1.shape)
            print("dw:", dW1)
            print("alpha*dw:", alpha*dW1)
            print("+")
            print("w:", weight)
            print("-=")
            weight -= alpha * dW1
            print("new Weight:", weight)
             
            dZ2 = dZ1
        
       
    
    
############ MAIN ###################




t = np.array([[1, 0, 0, 0],
              [1, 1, 1, 0],
              [1, 1, 0, 1],
              [1, 1, 1, 1], 
              [1, 0, 1, 1], 
              [1, 0, 1, 1]
              ]).T
t_out = np.array([[0,1,1,1,0,0]])

print("shape t:", t.shape , " t_out:", t_out.shape)
neuron_list = [4, 3, 2, 1]
layers = len(neuron_list)
print("NeuronList:", neuron_list, " layers:", layers )


## init
weights, biases = init_params(neuron_list)
    
        
print(" ***** GRADIENT DESCENT*******")
for i in range(1):
    ## forward feed
    zetas, acts = forward_feed(weights, biases, t)

    print("**** DONE FORWARD****")
    print("zetas:",len(zetas), " z:", zetas)
    print("acts:",len(acts), " a:", acts)

    ## back prop
    back_prop(weights, biases, zetas, acts, 0.15, t, t_out)
    
    
""" 
    
w2 = weights[2]
a1 = acts[1]
z1 = zetas[1]
a2 = acts[2]
z2 = zetas[2] 
print("w2:" , w2.shape, " a:", a1.shape , "z:" , z1.shape,"==> result a:", a2.shape, ":", z2.shape)
print("expected result:", t_out.shape )
dZ2 = t_out - a2
adj2 = dZ2 * sigmoid_deriv(a2)
print("dz2:", dZ2.shape, "=", t_out.shape, "-", a2.shape,  " adj2:", adj2.shape)

print("use a1:", a1.shape, " adjs2:", adj2.shape, " to adjust weight w2", w2.shape)
m2 = np.dot(adj2, a1.T)

print("flip a1:", a1.T.shape, " adjs2:", adj2.shape, "dot", m2.shape) 
print(w2 ," add ", m2)
w2 += m2
print(w2)
print("a1", a1)
print("adj", adj2)



"""
"""
w1=weights[0]
b1=biases[0]
z1=w1.dot(t)+b1
print("Shape w1:", w1.shape, "t:", t.shape, "z1:",z1.shape, " z1:", z1 )
a1 = ReLU(z1)
w2 = weights[1]
b2 = biases[1]
print("Shape w2:", w2.shape, " a1:", a1.shape , "b2:", b2.shape)
z2 = w2.dot(a1)+b2
"""
#dumpNpArray("w", weights)
#biases = [np.zeros(y)+1 for y in neuron_list[1:]]