import math
import numpy as np

# sigmod function for neuron activation values
def sigmoid(x):
    result = 1 / (1 + np.exp(-x))
    return result

# Class to instantiate network structure
class NeuralNetwork(object):
    hActivation = []

    def __init__(self, inputNodes, hiddenNodes, outputNodes):
        self.i = inputNodes
        self.h = hiddenNodes
        self.o = outputNodes

        # intialize random weights for both layers
        self.weights_1 = np.random.randint(1.0, size=(hiddenNodes, inputNodes))
        self.weights_2 = np.random.randint(1.0, size=(hiddenNodes, outputNodes))
        
        # initialize random biases 
        self.bias_1 = np.random.randint(10, size=(hiddenNodes, 1))
        self.bias_2 = np.random.randint(10, size=(outputNodes, 1))

    def guess(self, data):
        self.hActivation = sigmoid(self.weights_1.dot(data) + self.bias_1)
        return self.hActivation
        