import math
import numpy as np

# sigmod function for neuron activation values
def sigmoid(x):
    result = 1 / (1 + np.exp(-x))
    return result

# Class to instantiate network structure
class NeuralNetwork(object):
    hActivation = []

    # default learning rate for the network
    lr = 0.1

    def __init__(self, inputNodes, hiddenNodes, outputNodes, outputLabels):
        # initialize network structure
        self.i = inputNodes
        self.h = hiddenNodes
        self.o = outputNodes

        # labels for what each output neuron represents
        self.labels = outputLabels

        # intialize random weights for both layers
        self.weights_1 = np.random.uniform(0.0, 1.01, size=(hiddenNodes, inputNodes))
        self.weights_2 = np.random.uniform(0.0, 1.01, size=(outputNodes, hiddenNodes))
        
        # initialize random biases 
        self.bias_1 = np.random.randint(10, size=(hiddenNodes, 1))
        self.bias_2 = np.random.randint(10, size=(outputNodes, 1))

    # returns the neural network's guess for the correct output based on the data
    def guess(self, data):
        guess = 0

        self.InputData = data

        # calculate activation for hidden neurons
        self.hActivation = sigmoid(self.weights_1.dot(data) + self.bias_1)

        # calculate activation for output neurons
        self.oActivation = sigmoid(self.weights_2.dot(self.hActivation) + self.bias_2)
        
        # cycle through the outputs to find greatest activation
        for output in self.oActivation:
            if output > guess:
                guess = output
        
        # return best guess
        return guess

    # uses backpropagation to optimize the weights and biases
    def train(self, inputs, targets, tIterations):
        for i in range(0, tIterations):
            # start by letting the network guess
            result = self.guess(inputs)

            # compute error of guess
            error = targets - result
            
            # calculate and add adjustments for the weights
            gradient2 = self.oActivation*(1 - self.oActivation)*error
            delta_weights_2 = gradient2.dot(self.hActivation.transpose())
            self.weights_2 += delta_weights_2

            # adjust the bias accordingly
            self.bias_2 += gradient2

            # backpropagate error to the hidden layer
            errorh = self.weights_2.transpose().dot(error)

            # calculate and add adjustments for the weights
            gradient1 = self.hActivation*(1 - self.hActivation)*errorh
            delta_weights_1 = gradient1.dot(self.InputData.transpose())
            self.weights_1 += delta_weights_1

            # adjust bias accordingly
            self.bias_1 += gradient1
            