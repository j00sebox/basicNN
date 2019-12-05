from nn import NeuralNetwork
import numpy as np
import random

def main():
    # xor example
    xor = NeuralNetwork(2, 3, 2, [0, 1])

    # training data and the respective targets
    inp = [ [[1], [1]], [[0], [1]], [[0], [0]], [[1], [0]]  ]
    targets = [ [[1], [0]], [[0], [1]], [[1], [0]], [[0], [1]]  ]

    print("Training...")
    
    # train for 50000 cycles using stochastic gradient descent
    for i in range(0, 50000): 
        # get random data set
        r = random.randrange(0, 4) 
        xor.train(inp[r], targets[r])
    
    print("Completed training")

    print("Results: ")

    # output network guesses for all possible data
    t1 = xor.guess([[1], [1]])
    print('Test 1: ')
    print('0: ', t1[0])
    print('1: ', t1[1])

    t2 = xor.guess([[0], [1]])
    print('Test 2: ')
    print('0: ', t2[0])
    print('1: ', t2[1])

    t3 = xor.guess([[1], [0]])
    print('Test 3: ')
    print('0: ', t3[0])
    print('1: ', t3[1])

    t4 = xor.guess([[0], [0]])
    print('Test 4: ')
    print('0: ', t4[0])
    print('1: ', t4[1])


if __name__ == "__main__":
    main()