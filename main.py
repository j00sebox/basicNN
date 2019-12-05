from nn import NeuralNetwork
import numpy as np
import random

def main():
    # xor example
    xor = NeuralNetwork(2, 3, 2, [0, 1])

    inp = [ [[1], [1]], [[0], [1]], [[0], [0]], [[1], [0]]  ]
    t2 = [ [[1], [0]], [[0], [1]], [[1], [0]], [[0], [1]]  ]
    t1 = [ [[0]], [[1]] , [[0]], [[1]] ]
 
    for i in range(0, 100000):  
        xor.train(inp[i%4], t2[i%4])

    print(xor.guess([[1], [1]]))
    print(xor.guess([[0], [1]]))
    print(xor.guess([[1], [0]]))
    print(xor.guess([[0], [0]]))


if __name__ == "__main__":
    main()