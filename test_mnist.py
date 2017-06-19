import mnist_loader
import network

"""
Obtain 'training_data' and 'test_data' from load_data_wrapper()
"""
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

"""
784 neurons in input layer, 28 x 28 neurons
30 neurons in hidden layer
10 neurons in output layer
"""
net = network.Network([784, 30, 10])

"""
Epochs: 30
Mini-batch size: 10
Learning rate: 3.0	
"""
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
