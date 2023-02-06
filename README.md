# ANN_from_scratch
This repository aims to create a complete neural network from scratch.


This code is summarized with several functions as follows : 

- initialize_network(n_inputs, n_hidden, n_outputs)
- activate(weights, inputs)
- transfer(activation)
- forward_propagate(network, inputs)
- transfer_derivative(output)
- backward_propagate_error(network, expected)
- update_weights(network, row, l_rate)
- train_network(network, train, l_rate, n_epoch, n_outputs)
- predict(network, row)
- accuracy_metric(actual, predicted)
- ANN_backpropag(train_set, test_set, learning_rate, epochs)
- cross_validation_split(dataset, folds)
- kfoldCV(dataset, f=5, learning_rate=0.5, epochs=100)


Please note that it is a student project work. 
Not everything is necessarily optimised as it should be, so keep an eye on this fact.


