from nn import NeuralNetwork


def main():
    x = NeuralNetwork(2, 3, 2)
    print(x.i)
    print(x.h)
    print(x.o)
    print(x.sig)

if __name__ == "__main__":
    main()