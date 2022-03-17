from random import random
import math  # might be able to just import the exp function and not the whole library
from tqdm import tqdm

def network_create(input_count, hidden_count, output_count):
    '''Returns a completed network double nested list with dictionaries inside with random weights (as a list) for each neuron connection.'''
    network = []
    layer = []
    neuron = []

    # Create weights for hidden layers.
    # Each neuron needs weights for each input. Repeat this for the entire layer.
    for a in range(hidden_count):
        for b in range(input_count + 1):  # +1 for bias.
            neuron.append(random())
        layer.append({'weights':neuron})
        neuron = []
          
    network.append(layer)
    layer = []

    # Create weights for output neurons.
    for a in range(output_count):
        for b in range(hidden_count + 1):  # +1 for bias.
            neuron.append(random())
        layer.append({'weights':neuron})
        neuron = []

    network.append(layer)
    '''
    # Pre-set weights for testing.
    network =   [
                    [
                        {'weights': [0.5, -0.2, 0.5]},
                        {'weights': [0.1, 0.2, 0.3]}
                    ],
                    [
                        {'weights': [0.7, 0.6, 0.2]},
                        {'weights': [0.9, 0.8, 0.4]}
                    ]
                ]
    '''
    return network


def network_train(dataset, targets, number_of_iterations):
    ''''''
 
    for epoch in tqdm(range(number_of_iterations)):   
        for row_index, row in enumerate(dataset):
            forward_propogation(network, row)
            target = targets[row_index]
            backward_propogation(network, row, target)
            
            # output compared to targets
            # either calc error then backpropogate or can this be done during that stage?


def backward_propogation(network, row, target):
    network.reverse()

    for layer_index, layer in enumerate(network):
        for neuron_index, neuron in enumerate(layer):
            deltas = []

            # Error calculation.
            if (layer_index == 0):  # if output layer:
                error = target[neuron_index] - neuron['output']
                neuron['error'] = error
            else:  # Hidden layer, larger error calculation.
                sum = 0
                next_layer = network[layer_index - 1]  # Technically layer BEFORE in REVERSED network.
                for next_layer_neuron in next_layer:
                    sum += next_layer_neuron['weights'][neuron_index] * next_layer_neuron['error']
                error = neuron['output'] * (1 - neuron['output']) * ( sum )

            # First hidden layer needs input data to reference, otherwise use layer before.
            next_layer = row
            flag = True
            if (layer_index + 1) < len(network):
                next_layer = network[layer_index + 1]
                flag = False

            # Delta calculation.
            for next_layer_neuron in next_layer:
                if flag:
                    input_value = next_layer_neuron
                else:
                    input_value = next_layer_neuron['output']
                delta = learning_rate * error * input_value
                deltas.append(delta)
            
            delta = learning_rate * error * 1  # Add bias delta.
            deltas.append(delta)
            neuron['deltas'] = deltas
    
    # Update weights throughout the network.
    for layer in network:
        for neuron in layer:
            for w in range(len(neuron['weights'])):
                neuron['weights'][w] += neuron['deltas'][w]

    network.reverse()
    return network


def activation(inputs, weights):
    '''Return summed dot products of inputs and weights.'''
    net = weights[-1]  # Use bias as the starting net--> no input data to multiply with.
    for a in range(len(inputs)):
        net += inputs[a] * weights[a]
    return net    


def sigmoid(x):
    '''Returns a value between -1 and 1.'''
    return 1 / (1 + math.exp(-x))


def forward_propogation(network, row):
    '''Returns the outputs from the network for this epoch.'''
    input = row
    for layer_index, layer in enumerate(network):
        input_next = []
        for neuron in layer:
            output = activation(input, neuron['weights'])
            if (layer_index < len(network) - 1):  # Every layer but output uses sigmoid.
                output = sigmoid(output)  
            neuron['output'] = output
            input_next.append(neuron['output'])
        input = input_next
    # The last "inputs" will actually be the outputs from the output neurons.
    return input
        

def network_test(dataset):
    for row in dataset:
        output = forward_propogation(network, row)
        if (output[0] > output[1]):  # Only works for binary problems.
            output = 0
        else:
            output = 1
        print(f"input= {row} \t output= {output}")
    print('\n')


def parse_file(filename, dataset, targets):
    f = open(filename,"r")
    lines = f.readlines()
    for line in lines:
        temp = line.split('\t')[0].split()
        for a in range(len(temp)):
            temp[a] = int(temp[a])
        dataset.append(temp)

        temp = line.split('\t')[1].split()
        for a in range(len(temp)):
            temp[a] = int(temp[a])
        targets.append(temp)
    return


filepath = '/Volumes/ExternalBH/backpropogation/'
filenames = ['data-OR.txt', 'data-AND.txt', 'data-XOR.txt']
for filename in filenames:
    filename = filepath + filename
    dataset = []
    targets = []

    parse_file(filename, dataset, targets)

    input_count = len(dataset[0])
    hidden_count = 2
    output_count = 2
    number_of_iterations = 500000
    learning_rate = 0.1
    print(filename.split('/')[-1])
    network = network_create(input_count, hidden_count, output_count)
    network_train(dataset, targets, number_of_iterations)
    network_test(dataset)
