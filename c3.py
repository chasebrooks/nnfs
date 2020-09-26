import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# inputs = [[1, 2, 3, 2.5],
#           [2., 5., -1., 2],
#           [-1.5, 2.7, 3.3, -0.8]]
# weights = [[0.2, 0.8, -0.5, 1],
#            [0.5, -0.91, 0.26, -0.5],
#            [-0.26, -0.27, 0.17, 0.87]]
# biases = [2, 3, 0.5]
# weights2 = [[0.1, -0.14, 0.5],
#            [-0.5, 0.12, -0.33],
#            [-0.44, 0.73, -0.13]]
# biases2 = [-1, 2, -0.5]

# # inputs (3x4) * weights.T (4x3) = layer1_outputs (3x3)
# layer1_outputs = np.dot(inputs, np.array(weights).T) + biases

# # # layer1_outputs (3x3) * weights2.T (3x3) = layer2_outputs (3x3)
# # layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2


# spiral data
# X, y = spiral_data(samples=100, classes=3)
# plt.scatter(X[:,0], X[:,1], c=y, cmap='brg')
# plt.show()

class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        # Initialize weights = random and biases = 0
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        

    def forward(self, inputs):
        # calculate output values from inputs, weights, and biases
        self.output = np.dot(inputs, self.weights) + self.biases

# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Perform a forward pass of our training data through this layer
dense1.forward(X)

print(dense1.output[:5])
