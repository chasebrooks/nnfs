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

# Common loss class
class Loss:
    
    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):

        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        return data_loss

class Loss_CategoriclCrossentropy(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):
        
        # Number of samples in batch
        samples = len(y_pred)
        
        # Clip data to prevent division by 0
        # Clip both sides to not move mean either way
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values 
        # if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples), 
                y_true
                ]
        
        # Mask values: if one hot encoded
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])
class_targets = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 1, 0]])



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

# Create Loss Function
loss_function = Loss_CategoriclCrossentropy()

# Perform a forward pass of our training data through this layer
dense1.forward(X)

# Forward pass through activation function
# Takes in output from previous layer
activation1.forward(dense1.output)

# Forward pass of dense 2
dense2.forward(activation1.output)

# pass dense 2 through softmax activation function
activation2.forward(dense2.output)

# Print first few softmax output examples
print(activation2.output[:5])

# take output of activation function 2 and output loss
loss = loss_function.calculate(activation2.output, y)

# Print loss value
print('loss: ', loss)

# Calculate accuracy from output of activation2 and targets
# calculate values along first axis
predictions = np.argmax(activation2.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions==y)

# Print accuracy
print('acc:', accuracy)

