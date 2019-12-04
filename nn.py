import math

def sigmoid(x):
    result = 1 / (1 + math.exp(-x))
    return result


class NeuralNetwork:
    def __init__(self, inputNodes, hiddenNodes, outputNodes):
        self.i = inputNodes
        self.h = hiddenNodes
        self.o = outputNodes
        self.sig = sigmoid(200)