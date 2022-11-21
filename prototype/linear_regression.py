import numpy as np
import matplotlib.pyplot as plt

def generate_dataset_simple(beta, n, std_dev):
    # Generate x as an array of `n` samples which can take a value between 0 and 100
    #x = np.zeros(np.random() * 100, 1)
    x = np.random.normal(size=(n, 1))
    # Generate the random error of n samples, with a random value from a normal distribution, with a standard
    # deviation provided in the function argument
    e = np.random.randn(n, 1) * std_dev
    # Calculate `y` according to the equation discussed
    y = x * beta + e
    print("x:", x.shape, "y:", y.shape)
    return x, y
  
def get_basic_dataset():
    x = [1, 2, 3, 4.2, 5]
    y = [1.01, 2.3, 3.23, 4.11, 5.2]

    x = np.array(x)
    y = np.array(y)
    return x, y

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y)/Y.size

  
#https://www.projectpro.io/article/the-a-z-guide-to-gradient-descent-algorithm-and-its-variants/434

#simple linear regression
#x, y =  get_basic_dataset()
x, y = generate_dataset_simple(50, 50, 10)

m = 0
c = 0

learning_rate = 0.001
epochs = 500

#print("x:", x.shape, "y:", y.shape)

#y_pred = X*m + b
#y_diff = (y.reshape(-1,1) — y_pred)
#dm = -2*(np.sum(y_diff*X)/X.shape[0])
#db = -2*(np.sum(y_diff)/X.shape[0])
#m= m — alpha*dm
#b = b — alpha*db


#y = mx + c
#m= (y2–y1)/(x2-x1)
#c = y1 — x1*(y2-y1)/(x2-x1)


# run gradient descent algorithm
for i in range(epochs):
    y_pred = m*x + c
    
    # work out the error
    #dM = (-2/len(x)) * sum(x*(y - y_pred))
    #dC = (-2/len(x)) * sum(x - y_pred) # i think this is wrong
   
    ## other formular
    y_diff = y - y_pred
   
    dM = (-2/len(x)) * sum(x*y_diff)
    dC = (-2/len(x)) * sum(y_diff)
    
    m_old = m
    c_old = c
    # do adjustment
    m -= learning_rate * dM
    c -= learning_rate * dC
    
    if (i+1) % (epochs/10) == 0:
            print(f"Iteration: {i+1} / {epochs}")
            prediction = get_predictions(y_pred)
            print(f'{get_accuracy(prediction, y):.3%}')
    """
    print("[",i, "] ynew:", ynew)
    print("     m*x+c:", m*x ,
            "m:", "{:.4f}".format(m_old), "=>", "{:.4f}".format(m),
            "c:", "{:.4f}".format(c_old), "=>", "{:.4f}".format(c))
    """
    
print("After GradientDescent m=", m, "c=", c)



# Types of Gradient Descent
# 1. Batch Gradient Descent
# 2. Stochastic Gradient Descent
#3. Mini-Batch Gradient Descent  (is combine of 1,2

# tweak learning rate 
#using a small learning rate of the range 0.01 or even 0.001.
#y_pred = coef[0] + coef[1]*np.array(x)

yn = c + m*x
plt.scatter(x, y, marker="1", label="Dataset")
plt.plot(x, yn, color ="g", label='Prediction')
plt.legend(bbox_to_anchor=(1,1))
plt.xlabel('Input')
plt.ylabel('Output')
plt.show()  
 



