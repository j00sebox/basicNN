# basicNN

## Description

A very simple neural network library to help understand the fundamentals of neural networks.
Written in python. Uses backpropagation with stochastic gradient descent.

As an input the class takes the amount of input nodes, an array of amount of hidden layer nodes, the amount of output nodes, the learning rate and the weight decay. 

Currently there are 3 non-linearities that this class can use: sigmoid, softmax, and ReLU.

The train function handles all the backpropagation but only can handle one training example at a time so no batch training. 

The loss function used is mean squarred error.

To predict on a data point you can use the compute_neurons() function or the guess() function. These both take the training data point as an argument bu the compute neurons function returns the output values and the nodes while the guess function returns the best fitting label.

## demo.py

This shows off how to use the library. It trains the network to learn how to perform a 2-bit xor operation. It defines the data it needs and runs it through n times picking random data to train on. Then it tries to guess on all 4 possibilities.

The following are results using with two hidden layer 3 and 4 nodes respectively. The hidden node are using sigmoid activation while the output layer is using softmax.

Results after 10 training examples:

![10ex](/screenshots/10ex.PNG)

As you can see it doesn't do very well after only 10 examples.

Results after 100 training examples:

![100ex](/screenshots/100ex.PNG)

It is still not good but it was able to adjust itself based on new data.

Results after 1000 training examples:

![1000ex](/screenshots/1000ex.PNG)

Now it is correctly classifying all but one test but it is still not very confident.

Results after 10000 training examples:

![10kex](/screenshots/10kex.PNG)

It classifies all the tests correctly and it is very confindent in the outcome.

Now of course this is a very basic example and of course in real world applications the datasset and the problem won't be so trivial but it is still interesting to see how all this works. 
