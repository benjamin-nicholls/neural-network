# random not used in this implementation with given weights but functionality is present.
from random import randrange  # Used for random weights in network creation.
from math import exp  # Used for error calculations.
from tqdm import tqdm  # Used for the epoch progress bar.
from matplotlib import pyplot as plt  # Used to plot errors.

def main() -> None:
    # Customisable variables.
    epoch_count = 200
    learning_rate = 0.1
    hidden_layers_neuron_list = [3]  # [3] is one hidden layer of 3 neurons. [4,4,3] is three hidden layers.
    filenames = ['data-assignment.txt', 'data-assignment-test.txt']

    ###### Random weights are overwritten in network_create() with given assignment weights. #####

    try:
        print('Enter filepath to data files (blank for current directory):')
        filepath = input('>>> ')
        # Append trailing / or \ for Mac/Windows, respectively.
        if '/' in filepath: 
            if (filepath[-1] != '/'): filepath += '/'
        elif '\\' in filepath:
            if (filepath[-1] != '\\'): filepath += '\\'
    except:
        print('Error in filepath. Setting to working directory.')
        filepath = ''


    dataset = []
    targets = []
    for filename in filenames:
        filename = filepath + filename

        if 'test' in filename:
            dataset, targets = parse_file(filename, dataset, targets, True)
            network_test(network, dataset)
            print_softmax(network)
        else:
            dataset, targets = parse_file(filename, dataset, targets, False)

            input_count = len(dataset[0])
            output_count = len(targets[0])

            layers_node_count = [input_count]
            for a in hidden_layers_neuron_list:
                layers_node_count.append(a)
            layers_node_count.append(output_count)

            print(f'\nTraining:\nLayers: {layers_node_count}, number of epochs: {epoch_count}, learning rate: {learning_rate}.\n')
            network = network_create(layers_node_count)
            network_train(network, dataset, targets, epoch_count, learning_rate, filename)
            # print_network_readable(network, epoch_count)
            network_test(network, dataset)
            
            print_softmax(network)
    return


def network_create(layers_node_count: list) -> list:
    '''Returns a completed network double nested list with dictionaries inside with random weights (as a list) for each neuron connection.'''
    # Dynamic creation of a network with random weights with any number of hidden layers.
    network = []
    layer = []
    neuron = []
    for layer_index, network_layer in enumerate(layers_node_count):
        if (layer_index == 0 or layer_index == len(layers_node_count)): continue  # Input layer has no weights.
        for neuron_count in range(network_layer):
            for connection_count in range(layers_node_count[layer_index - 1] + 1):  # +1 for bias.
                neuron.append(randrange(-5, 5)/100)
            layer.append({'weights':neuron})
            neuron = []
        network.append(layer)
        layer = []

    # Assignment weights - overwrites random weights from above.
    if True:
        network =   [
                        [
                            {'weights': [0.74, 0.80, 0.35, 0.90]},
                            {'weights': [0.13, 0.40, 0.97, 0.45]},
                            {'weights': [0.68, 0.10, 0.96, 0.36]}
                        ],
                        [
                            {'weights': [0.35, 0.50, 0.90, 0.98]},
                            {'weights': [0.80, 0.13, 0.80, 0.92]}
                        ]
                    ]

    return network


def network_train(network: list, dataset: list, targets: list, epoch_count: int, learning_rate: float, fileName: str) -> list:
    '''Trains the network using the dataset for the epoch count. Forward and then backpropogation. Errors are graphed. Returns network.'''
    error_list = []
    for epoch in tqdm(range(epoch_count)):
        error_squared = 0
        for row_index, row in enumerate(dataset):
            forward_propogation(network, row)
            target = targets[row_index]
            backward_propogation(network, row, target, learning_rate)
            error_squared += calculate_error_squared(network)
        error_list.append(error_squared)

    if '/' in fileName: 
        filename_temp = fileName.split('/')[-1]
    elif '\\' in fileName:
        filename_temp = fileName.split('\\')[-1]
    else:
        filename_temp = fileName
    name =  filename_temp + '. Epochs= ' + str(epoch_count) + '. L= ' + str(learning_rate)
    plot_learning_curve(error_list, name)
    return network


def forward_propogation(network: list, row: list) -> list:
    '''Returns the outputs from the network for this epoch.'''
    input = row
    for layer_index, layer in enumerate(network):
        input_next = []
        for neuron in layer:
            output = activation(input, neuron['weights'])
            if (layer_index < len(network) - 1):  # Output layer does not use sigmoid.
                output = sigmoid(output)  
            neuron['output'] = output
            input_next.append(neuron['output'])
        input = input_next
    return input  # The last "inputs" will actually be the outputs from the output neurons.
        

def backward_propogation(network: list, row: list, target: int, learning_rate: float) -> list:
    '''Backpropogation for the network. Network updates weights, errors, and deltas. Returns network.'''
    network.reverse()

    for layer_index, layer in enumerate(network):
        for neuron_index, neuron in enumerate(layer):
            deltas = []

            # Error calculation.
            if (layer_index == 0):  # if output layer:
                neuron['error'] = target[neuron_index] - neuron['output']
            else:  # Hidden layer, larger error calculation.
                sum = 0
                next_layer = network[layer_index - 1]  # Technically layer BEFORE in REVERSED network.
                for next_layer_neuron in next_layer:
                    sum += next_layer_neuron['weights'][neuron_index] * next_layer_neuron['error'] 
                neuron['error'] = neuron['output'] * (1 - neuron['output']) * ( sum )

            # First hidden layer needs input data to reference, otherwise use layer before.
            next_layer = row
            flag_first_hidden_layer = True
            if (layer_index + 1) < len(network):
                next_layer = network[layer_index + 1]
                flag_first_hidden_layer = False

            # Delta calculation.
            for next_layer_neuron in next_layer:
                if flag_first_hidden_layer:
                    input_value = next_layer_neuron
                else:
                    input_value = next_layer_neuron['output']
                delta = learning_rate * neuron['error'] * input_value
                deltas.append(delta)
            
            delta = learning_rate * neuron['error'] * 1  # Add bias delta.
            deltas.append(delta)
            neuron['deltas'] = deltas

    # Update weights throughout the network.
    for layer in network:
        for neuron in layer:
            for w in range(len(neuron['weights'])):
                neuron['weights'][w] += neuron['deltas'][w]

    network.reverse()
    return network


def network_test(network: list, dataset: list) -> list:
    '''Forward propagates through the network.'''
    print('\nTesting:')
    for row in dataset:
        output = forward_propogation(network, row)
        for o_index, o in enumerate(output):
            if o == max(output): max_output_index = o_index
        print(f'input= {row} \t output= {max_output_index}')
    return network


def activation(inputs: list, weights: list) -> float:
    '''Return summed dot products of inputs and weights.'''
    net = weights[-1]  # Use bias as the starting net--> no input data to multiply with.
    for a in range(len(inputs)):
        net += inputs[a] * weights[a]
    return net    


def sigmoid(x: float) -> float:
    '''Returns a value between -1 and 1.'''
    return 1 / (1 + exp(-x))


def softmax_function(layer: list) -> list:
    '''Calculates the probability each output neuron has of being activated. Returns this as a list.'''
    sum = 0
    softmax_list = []
    for neuron_index, neuron in enumerate(layer):
        sum += exp(neuron['output'])
    for neuron_index, neuron in enumerate(layer):
        softmax = exp(neuron['output']) / sum
        softmax_list.append(softmax)
    return softmax_list


def calculate_error_squared(network: list) -> float:
    '''Sums squared errors from output layer of the network. Returns average.'''
    error_squared = 0
    for output_neuron in network[-1]:
        error_squared += output_neuron['error']**2
    error_squared = error_squared / len(network[-1])
    return error_squared


def parse_file(filename: str, dataset: list, targets: list, isThisTestData: bool) -> tuple:
    '''Parses file and returns dataset and targets nested lists.'''
    try:
        f = open(filename,'r')
        lines = f.readlines()
    except FileNotFoundError:
        print(f'\n\nFile not found in current working directory or at location: {filename} \n\n')
        exit()
    
    dataset = []
    targets = []

    for line in lines:
        temp = line.split('\t')[0].split()
        for a in range(len(temp)):
            temp[a] = float(temp[a])
        dataset.append(temp)
        if (not isThisTestData):  # Test data has no targets in file.
            temp = line.split('\t')[1].split()
            for a in range(len(temp)):
                temp[a] = int(temp[a])
            targets.append(temp)

    print('\nUsing data from: ', filename)
    return dataset, targets


def plot_learning_curve(errors: list, name: str) -> None:
    '''Plots a graph of errors squared vs epoch. Graph is a popup. No return.'''
    x_data = []
    y_data = []
    x_data.extend(epoch for epoch in range(len(errors)))#epoch for epoch, data in enumerate(errors))
    y_data.extend(error for error in errors)
    fig, ax = plt.subplots()
    fig.suptitle(name)
    ax.set(xlabel='Epoch', ylabel='Squared Error')
    ax.plot(x_data, y_data, 'tab:green')
    plt.show()
    return


def print_softmax(network: list) -> None:
    '''Prints the softmax values of the output layer neurons. No return.'''
    softmax = softmax_function(network[-1])
    print('softmax=', softmax)
    return


def print_network_readable(network: list, epoch: int) -> None:
    '''Prints the network by neuron. Dictionary entries printed on one line each. No return.'''
    print('EPOCH=', epoch)
    for layer_index, layer in enumerate(network):
        for neuron_index, neuron in enumerate(layer):
            print('layer=', layer_index, ', neuron=', neuron_index)
            for key, values in neuron.items():
                print('\t\t', key, ': ', values)
            print('\n')
        print('\n')
    return


if __name__ == '__main__':
    main()
