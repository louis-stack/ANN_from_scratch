import pandas as pd
import numpy as np
from math import exp
import matplotlib.pyplot as plt
from random import seed, randrange
from random import random
from mpl_toolkits.mplot3d import Axes3D

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

#activation = sum(weight_i * input_i) + bias
# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

#output = 1 / (1 + e^(-activation))
# Transfer neuron activation (Sigmoid Threshold function)
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))


# Forward propagate input to a network output
def forward_propagate(network, inputs):
	actual_inputs = inputs
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], actual_inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		actual_inputs = new_inputs
	return actual_inputs

#derivative = output * (1.0 - output)
# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
        # Hidden layers
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
        # Outputs 
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(neuron['output'] - expected)

        for j in range(len(layer)):
            #delta weight calculation
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


#weight = weight - learning_rate * error * input
# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] -= l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
	list_error = list()
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			output = forward_propagate(network, row)

			expected = int(row[-1])

			sum_error += (expected-output[0])**2
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		list_error.append(sum_error)
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
	return list_error




# Make a prediction with a network
def predict(network, row):
	outputs = forward_propagate(network, row)

	if outputs[0] < 0.5:
		return 0
	else :
		return 1
	 

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

def ANN_backpropag(train_set, test_set, learning_rate, epochs) :

	n_inputs = 19
	n_outputs = 1
	n_neurons = 6
	network = initialize_network(n_inputs, n_neurons, n_outputs)
	list_error = train_network(network, train_set, learning_rate, epochs, n_outputs)
	for layer in network:
		print(layer)

	# Predict + score
	compteur = 0
	predicted = list()
	actual = list()

	for row in test_set:
		compteur += 1
		prediction = predict(network, row)
		predicted.append(prediction)
		actual.append(row[-1])
		print('Expected=%d, Got=%d' % (row[-1], prediction))

	accuracy = accuracy_metric(actual, predicted)
	#print("\nAccuracy : %0.6f\n" % (accuracy))
	return accuracy, list_error

def cross_validation_split(dataset, folds):
        dataset_split = []
        df_copy = dataset
        fold_size = int(df_copy.shape[0] / folds)
        
        # for loop to save each fold
        for i in range(folds):
            fold = []
            # while loop to add elements to the folds
            while len(fold) < fold_size:
                # select a random element
                r = randrange(df_copy.shape[0])
                # determine the index of this element 
                index = df_copy.index[r]
                # save the randomly selected line 
                fold.append(df_copy.loc[index].values.tolist())
                # delete the randomly selected line from
                # dataframe not to select again
                df_copy = df_copy.drop(index)
            # save the fold     
            dataset_split.append(np.asarray(fold))
            
        return dataset_split 

def kfoldCV(dataset, f=5, learning_rate=0.5, epochs=100):
	data=cross_validation_split(dataset,f)
	result=[]
	# determine training and test sets 
	for i in range(f):
		r = list(range(f))
		r.pop(i)
		for j in r :
			if j == r[0]:
				cv = data[j]
		else:    
			cv=np.concatenate((cv,data[j]), axis=0)

		acc, list_error = ANN_backpropag(cv.tolist(), data[i], learning_rate, epochs)
		result.append(acc)

	return (sum(result) / f), list_error

#Set Seed
seed(1)

#Test  training

dataset = pd.read_csv("data/Diabetic.csv", sep=';')
dataset_shuffled = dataset.sample(frac=1)

# Normalization min max
normalized_df=(dataset_shuffled-dataset_shuffled.min())/(dataset_shuffled.max()-dataset_shuffled.min())

#Initial dataset
# Data for three-dimensional scattered points*

fig = plt.figure(figsize = (10, 8))
ax = plt.axes(projection='3d')
ax.scatter3D(normalized_df.iloc[:,11], normalized_df.iloc[:,13], normalized_df.iloc[:,15])
ax.set_xlim3d(0, 1)
ax.set_ylim3d(0, 1)
ax.set_zlim3d(0, 1)

ax.set_xlabel('MA result1')
ax.set_ylabel('MA result3')
ax.set_zlabel('MA result5')
ax.set_title('MA result scatter')

#TEST WICH PARAMETERS ARE THE BEST 
#RESULT : learning_rate = 0.5, epochs = 800
"""
number_of_folds = 10
test_set = (normalized_df[1037:1151])
list_learning_rate_values = [0.01, 0.1, 0.5, 1, 5]
list_epochs_values = [5, 20, 100, 500, 800, 1000]
results = list()

for i in range(len(list_learning_rate_values)):
	for j in range(len(list_epochs_values)):

		result, list_error_valid = kfoldCV(normalized_df[0:1036], number_of_folds, list_learning_rate_values[i], list_epochs_values[j])
		
		predicted, list_error_test = ANN_backpropag(normalized_df[0:1036].values.tolist(), test_set.values.tolist(), list_learning_rate_values[i], list_epochs_values[j])
		results.append([i, j, result, predicted, result + predicted])

print(results)

"""

learning_rate = 0.5
epochs = 800
number_of_folds = 10

result, list_error_valid = kfoldCV(normalized_df[0:1036], number_of_folds, learning_rate, epochs)

test_set = (normalized_df[1037:1151])

predicted, list_error_test = ANN_backpropag(normalized_df[0:1036].values.tolist(), test_set.values.tolist(), learning_rate, epochs)

list_epochs = list()
for i in range(epochs):
    list_epochs.append(i + 1)

fig = plt.figure(figsize = (10, 8))
plt.plot(list_epochs, list_error_valid)
plt.title("validation accuracy during the learning process")
plt.xlabel("epochs")
plt.ylabel("error")
plt.show()

print("crossval result : ", result)
print("testing result : ", predicted)
