import numpy as np


def normalize_input(input_val):
    x = np.array(input_val)
    mean = np.mean(x)
    std = np.std(x)

    normalized = (x - mean) / std
    norm_min = np.min(normalized)
    if norm_min < 0:
        normalized -= norm_min
    return normalized

def normalize_layer(layer):
    # assuming we have a list of objects with a 'temp' property
    # extract the 'temp' values from the list
    act_values = [neuron.temp for neuron in layer]

    # calculate mean and standard deviation
    mean = np.mean(act_values)
    std_dev = np.std(act_values)

    # check if the standard deviation is zero
    if std_dev == 0:
        # if standard deviation is zero, then all the values in the list are the same
        # and we cannot normalize the values
        return

    # loop through the list and normalize the 'temp' values
    for neuron in layer:
        if neuron.temp < 0:
            # if the value is negative, normalize it using the absolute value of the mean and standard deviation
            neuron.temp = -((abs(neuron.temp) - mean) / std_dev)
        elif neuron.temp == 0:
            # if the value is zero, set it to the mean value
            neuron.temp = mean
        else:
            # if the value is positive, normalize it using the mean and standard deviation
            neuron.temp = (neuron.temp - mean) / std_dev


def normalize_layer_weights(layer):
    for neuron in layer:
        # extract the parent_values for this neuron
        parent_values = [p[1] for p in neuron.parents]

        # calculate mean and standard deviation for this neuron
        mean = np.mean(parent_values)
        std_dev = np.std(parent_values)

        # check if the standard deviation is zero for this neuron
        if std_dev == 0:
            # if standard deviation is zero, then all the values in the list are the same
            # and we cannot normalize the values
            continue

        # loop through the list of parents and normalize the values for this neuron
        for i, p in enumerate(neuron.parents):
            if p[1] < 0:
                # if the value is negative, normalize it using the absolute value of the mean and standard deviation
                neuron.parents[i][1] = -((abs(p[1]) - mean) / std_dev)
            elif p[1] == 0:
                # if the value is zero, set it to the mean value
                neuron.parents[i][1] = mean
            else:
                # if the value is positive, normalize it using the mean and standard deviation
                neuron.parents[i][1] = (p[1] - mean) / std_dev
