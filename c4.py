import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data


class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        # Initialize weights = random and biases = 0
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        

    def forward(self, inputs):
        # calculate output values from inputs, weights, and biases
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:

    # Forward pass
    def forward(self, inputs):
        # Calculate output values from input: y if X>0, 0 ow
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    # accept non-normalized inputs from hidden layer activation functions
    # output a probability distribution for each class
    # Forward pass
    def forward(self, inputs):
        # Get unnormalized probablilities from inputs: subtract the largest number before exponentiation to avoid exploding exponentiation... 
        # all negative values are between [0,1] when exponentiated 
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize exp_values for each sample
        probablities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probablities



#Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Create ReLU activation (to be used with Dense layer)
activation1 = Activation_ReLU()

# Create a second Dense layer with 3 input features (taken from previous layer's outputs)
# and 3 outputs
dense2 = Layer_Dense(3, 3)

# Create Softmax Activation (for dense2)
activation2 = Activation_Softmax()

# Perform a forward pass of our training data through this layer
dense1.forward(X)

# Forward pass through activation function
# Takes in output from previous layer
activation1.forward(dense1.output)


# Forward pass of dense 2
dense2.forward(activation1.output)

# pass dense 2 through softmax activation function
activation2.forward(dense2.output)

print(activation2.output[:5])
