
import numpy as np

print("Hello Perceptron Vanilla Prototype")

#need sudo pacman -S python python-numpy python-matplotlib
# to run python nn-vanilla.py
# Activation function sigmoid
def sigmoid(x):
        return 1 / (1 + np.exp(-x))

# Sigmoid derivative for backpropagation
def sigmoid_deriv(x):
        return x*(1-x)

# The training set, with 4 examples consisting of 3 input values and 1 output value
training_inputs = np.array([    [0, 0, 1],
                                [1, 1, 1],
                                [1, 0, 1],
                                [0, 1, 1] ])

#transpose it to trun into 4X1 matrix
training_outputs = np.array([[0,1,1,0]]).T

# set to have the same random number as clips.
np.random.seed(1)

#create 3X1 matrix of random weights between -1 to 1
synaptic_weights = 2 * np.random.random((3, 1)) -1

print ("Random Starting Synaptic Weights: ")
print(synaptic_weights)

# test with 1, then put it to a bigger range 20000 result same as tutorial.
# then try 50000, as the iteration goes to infinity. results approoach 1.

for iteration in range(50000):
    input_layer = training_inputs
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    #calculate error and adjustments for backpropagation
    error = training_outputs - outputs
    adjustments = error * sigmoid_deriv(outputs)
    synaptic_weights += np.dot(input_layer.T, adjustments)

print("Synatic weights after training")
print(synaptic_weights)
print("Output after training:")
print(outputs)

# test input expect test output of 0.99xxx
test_input = np.array([1,0,0])
test_output = sigmoid(np.dot(test_input, synaptic_weights))

print("Output test_input data:", test_input, test_output)

