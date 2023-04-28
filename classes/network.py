import math
import random

from app.classes.neuron import Neuron
import statistics
import multiprocessing

from app.utils.normalizers import normalize_layer
import random


def random_nonzero():
    # generate a random float between -1 and 1
    # if the value is 0, generate a new value until it is nonzero
    value = 0
    while value == 0:
        value = random.uniform(-1, 1)
    return value

def calculate_activation(neuron):
    neuron.calculate_act()


class Network:
    def __init__(self, inputs, image, label, outputs):
        self.inputs = inputs
        self.image = image
        self.label = label
        self.outputs = outputs
        self.neuronChainMap = []
        self.input_layer = []
        self.hidden_layer = []
        self.output_layer = {}
        self.trained = 1
        self.create_input_layer()
        self.create_hidden_layer(self.input_layer)
        self.create_output_layer()

    def create_input_layer(self):
        for i in range(0, self.inputs):
            self.input_layer.append(Neuron())

    def create_hidden_layer(self, prev_layer):
        for i in range(int(statistics.mean([len(prev_layer), len(self.outputs)]))):
            new_neuron = Neuron()
            for neuron in prev_layer:
                new_neuron.add_parent(neuron, random_nonzero())
            self.hidden_layer.append(new_neuron)

    def create_output_layer(self):
        for output in self.outputs:
            new_neuron = Neuron()
            for neuron in self.hidden_layer:
                new_neuron.add_parent(neuron, random_nonzero())
            self.output_layer[output] = new_neuron

    def forward_input(self):
        for i, neuron in enumerate(self.input_layer):
            neuron.activation = self.image[i]
        # with multiprocessing.Pool() as pool:
        #     try:
        #         pool.map(self.calculate_layer_act, self.hidden_layers)
        #     finally:
        #         pool.close()
        #         pool.join()

    def get_result(self):
        # for key, value in self.output_layer.items():
        for neuron in self.hidden_layer:
            neuron.calculate_temp()
        normalize_layer(self.hidden_layer)

        for neuron in self.hidden_layer:
            neuron.calculate_act()

        for key, value in self.output_layer.items():
            value.calculate_temp()
        normalize_layer(self.output_layer.values())

        for key, value in self.output_layer.items():
            value.calculate_act()

    def cleanup(self):
        for neuron in self.input_layer:
            neuron.temp = 0
            neuron.activation = 0

        for neuron in self.hidden_layer:
            neuron.temp = 0
            neuron.activation = 0

        for key, value in self.output_layer.items():
            value.temp = 0
            value.activation = 0

    # def back_propagate(self, layer):
    #     desired = 1
    #     for neuron in layer:
    #         for parent in neuron.parents:
    #             d_error = 2 * (neuron.activation - desired)
    #             d_aw = 0
    #             if neuron.activation == 0:
    #                 d_aw = 0.5
    #             if parent[1] == 0:
    #                 d_aw = 10000000
    #             if neuron.activation != 0 and parent[1] != 0:
    #                 d_aw = math.log(((1 - neuron.activation) / neuron.activation)) / parent[1]
    #             correction = d_error * d_aw
    #             parent[1] += correction/self.trained
    #     if self.trained < 20:
    #         self.trained += 1

    def back_propagate_from_output(self):
        for key, value in self.output_layer.items():
            if key == str(self.label):
                desired = 1
            else:
                desired = 0
            dc_da = value.activation - desired  # Derivative of the cost with respect to activation
            da_dz = (value.activation - math.pow(value.activation, 2))  # Derivative of act with respect to weighted sum
            # dz_dw = value.calculate_parent_sum_weight()  # Derivative of weighted sum with respect to weights
            chain = dc_da * da_dz
            self.neuronChainMap.append(chain)
            for ni, neuron in enumerate(value.parents):
                da_dw = neuron[0].activation
                correction = dc_da * da_dw # Derivative of cost with respect to weight
                neuron[1] -= 0.01 * correction

    def process_hidden_layer(self):
        for neuron in self.hidden_layer:
            da_dz = neuron.activation * (1 - neuron.activation)  # Derivative of activation with respect to weighted sum
            delta = sum([self.neuronChainMap[k] * neuron.parents[k][1] for k in range(len(self.outputs))]) * da_dz

            for ni, p_neuron in enumerate(neuron.parents):
                da_dw = p_neuron[0].activation
                correction = delta * da_dw
                p_neuron[1] -= 0.01 * correction

    def try_image(self, image, label):
        self.image = image
        self.label = label
        self.forward_input()
        self.get_result()
        results = []
        for key, value in self.output_layer.items():
            results.append((key, value.activation))
            # print(key)
            # print(value.activation)
        # sort the list of tuples by the second value (probability)
        sorted_tuples = sorted(results, key=lambda x: x[1], reverse=True)

        # print the sorted tuples with labels
        for i, t in enumerate(sorted_tuples):
            if t[0] == str(self.label):
                print(f">Number: {t[0]}, Probability: {t[1]}")
            else:
                print(f"Number: {t[0]}, Probability: {t[1]}")

    def train(self, image, label):
        self.image = image
        self.label = label
        self.forward_input()
        self.get_result()
        self.neuronChainMap = []
        self.back_propagate_from_output()
        # for chain in self.neuronChainMap:
        self.process_hidden_layer()
        self.cleanup()
        # for layer in self.hidden_layers:
        #     self.back_propagate(layer)
        # with multiprocessing.Pool() as pool:
        #     pool.starmap(self.process_neuron, [(neuron, ni, layerIndex, d_error) for ni, neuron in enumerate(layer)])
        # for ci, chain in enumerate(self.neuronChainMap[0]):
        #     first_layer = True
        #     for li, layer in enumerate(self.hidden_layers):
        #         if first_layer:
        #             first_layer = False
        #             [self.process_neuron(neuron, 0, ci) for ni, neuron in enumerate(layer)]
        #         else:
        #             [self.process_neuron(neuron, li, ni) for ni, neuron in enumerate(layer)]
    # def process_layer(self, layer, layerIndex, d_error):
    #     with multiprocessing.Pool() as pool:
    #         pool.starmap(self.process_neuron, [(neuron, ni, layerIndex, d_error) for ni, neuron in enumerate(layer)])
