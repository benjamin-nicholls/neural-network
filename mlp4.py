from random import random
import math  # might be able to just import the exp function and not the whole library
from tqdm import tqdm

def main():
    filepath = '/Volumes/ExternalBH/backpropogation/'
    filenames = ['test-data.txt']#, 'data-OR.txt', 'data-AND.txt', 'data-XOR.txt']
    for filename in filenames:
        filename = filepath + filename
        dataset = []
        targets = []
        parse_file(filename, dataset, targets)
        input_count = len(dataset[0])
        output_count = len(targets[0])  # how to find this out from data
        layers_node_count = [input_count,2,output_count]
        epoch_count = 10
        learning_rate = 0.1
        print(f'\nLayers: {layers_node_count}, number of epochs: {epoch_count}, learning rate: {learning_rate}.\n')

        print(filename.split('/')[-1])
        network = network_create(layers_node_count)
        network_train(network, dataset, targets, epoch_count, learning_rate)
        network_test(network, dataset)
        #print(network)

def network_create(layers_node_count):
    '''Returns a completed network double nested list with dictionaries inside with random weights (as a list) for each neuron connection.'''    

    network = []
    layer = []
    neuron = []
    for layer_index, network_layer in enumerate(layers_node_count):
        if (layer_index == 0 or layer_index == len(layers_node_count)): continue  # Input layer has no weights.
        for neuron_count in range(network_layer):
            for connection_count in range(layers_node_count[layer_index - 1] + 1):  # +1 for bias.
                neuron.append(random())
            layer.append({'weights':neuron})
            neuron = []
        network.append(layer)
        layer = []

    if True:
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

    return network


def network_train(network, dataset, targets, epoch_count, learning_rate):
    ''''''
    for epoch in tqdm(range(epoch_count)):   
        for row_index, row in enumerate(dataset):
            forward_propogation(network, row)
            target = targets[row_index]
            backward_propogation(network, row, target, learning_rate)
        print('\nepoch= ', epoch+1)
        print(network)


def backward_propogation(network, row, target, learning_rate):
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
                neuron['error'] = error

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
        

def network_test(network, dataset):
    for row in dataset:
        output = forward_propogation(network, row)
        if (output[0] > output[1]):  # Only works for binary problems.
            output = 0
        else:
            output = 1
        print(f'input= {row} \t output= {output}')
    print('\n')
    return


def parse_file(filename, dataset, targets):
    try:
        filename_temp = filename.split('/')[-1]
        f = open(filename_temp,'r')
        lines = f.readlines()
    except FileNotFoundError:
        try:
            f = open(filename,'r')
            lines = f.readlines()
        except:
            print(f'\n\nFile \'{filename_temp}\' not found in current working directory or at location: {filename} \n\n')
            lines = ['0 0\t0 0']
    
    for line in lines:
        temp = line.split('\t')[0].split()
        for a in range(len(temp)):
            temp[a] = int(temp[a])
        dataset.append(temp)

        temp = line.split('\t')[1].split()
        for a in range(len(temp)):
            temp[a] = int(temp[a])
        targets.append(temp)
    return dataset, targets


if __name__ == '__main__':
    main()