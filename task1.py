"""
Created on 19 Feb 2017
@author: YoonSeong Yong
"""
import pandas
import numpy as np
import network

def load_data():
	"""
	Return the 'data/iris.data' as a tuple containing the training data,
	and the test data.

	The 'training_data' is returned as a tuple with two entries.
	The first entry is a numpy ndarray with 120 entries (row 1~40, 
	row 51~90 and row 101~140) 
	Each entry is a numpy ndarray with 4 values, representing sepal 
	length in cm, sepal width in cm, petal length in cm and petal width 
	in cm.

	The second entry in the 'training_data' tuple is a numpy ndarray
	containing 120 entries (row 1~40, row 51~90 and row 101~140). 
	Those entries are just the digit values (0...2) to represent '0' as 
	Iris Setosa, '1' as Iris Versicolor and '2' as Iris Virginica    

	The 'test_data' is similar, except it contains only 30 entries 
	(row 41~50, row 91~100, row 141~150)
	"""    
	df = pandas.read_csv('data/iris.data', header=None)
	y = df.iloc[0:df.shape[0], 4].values
	y = np.where(y == 'Iris-setosa', 0, y)
	y = np.where(y == 'Iris-versicolor', 1, y)
	y = np.where(y == 'Iris-virginica', 2, y)
	x = df.iloc[0:df.shape[0], 0:4].values
	x = tuple(x)
	y = tuple(y)
	training_inputs = x[0:40] + x[50:90] + x[100:140]
	training_results = y[0:40] + y[50:90] + y[100:140]
	training_data = (training_inputs, training_results)
	test_inputs = x[40:50] + x[90:100] + x[140:150]
	test_results = y[40:50] + y[90:1000] + y[140:150]
	test_data = (test_inputs, test_results)
	return (training_data, test_data)

def load_data_wrapper():
	"""
	Return a tuple containing '(training_data, test_data)' based on 
	'load_data', but the format is more convenient for use in 
	implementation of neural networks.

	In particular, 'training_data' is a list containing 120
	2-tuples '(x, y)'.  'x' is a 4-dimensional numpy.ndarray
	containing the features of iris. 'y' is a 3-dimensional
	numpy.ndarray representing the unit vector corresponding to the
	correct digit for 'x'.

	'test_data' is list containing 30 2-tuples '(x, y)'.  
	In each case, 'x' is a 4-dimensional numpy.ndarry containing the 
	features of iris, and 'y' is the corresponding classification, 
	i.e., the digit values (integers) corresponding to 'x'.
	"""
	tr_d, te_d = load_data()
	training_inputs = [np.reshape(x, (4, 1)) for x in tr_d[0]]
	training_results = [vectorized_result(y) for y in tr_d[1]]
	training_data = zip(training_inputs, training_results)
	test_inputs = [np.reshape(x, (4, 1)) for x in te_d[0]]
	test_data = zip(test_inputs, te_d[1])
	return (training_data, test_data)

def vectorized_result(j):
	"""
	Return a 3-dimensional unit vector with a 1.0 in the jth
	position and zeroes elsewhere.  This is used to convert a digit
	(0..2) into a corresponding desired output from the neural
	network.
	"""
	e = np.zeros((3, 1))
	e[j] = 1.0
	return e

if __name__ == "__main__":
    """
    Obtain 'training_data' and 'test_data' from load_data_wrapper()

    4 neurons in input layer
    10 neurons in hidden layer1
    10 neurons in hidden layer2
    3 neurons in output layer

    Epochs: 30
    Mini-batch size: 10
    Learning rate: 0.5	
    """
    training_data, test_data = load_data_wrapper()
    print "Training data size = ", len(training_data)
    print "Test data size = ", len(test_data)
    net = network.Network([4, 10, 10, 3])
    net.SGD(training_data, 30, 10, 0.5, test_data=test_data)
