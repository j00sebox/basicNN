from nn import NeuralNetwork
import numpy as np


def main():
    x = NeuralNetwork(2, 3, 2, [0, 1])
    # print(x.i)
    # print(x.h)
    # print(x.o)

    # # weight matrixes
    # print(x.weights_1)
    # print(x.weights_2)

    # # bias matrixes
    # print(x.bias_1)
    # print(x.bias_2)

    i = np.random.randint(1.0, size=(x.i, 1))
    t = [[1], [0]]
    print(x.train(i, t, 1000))



if __name__ == "__main__":
    main()