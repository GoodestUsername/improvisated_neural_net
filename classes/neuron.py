from scipy.special import expit


class Neuron:
    def __init__(self, activation=0):
        self.activation = activation
        # self.bias = 0
        self.temp = 0
        self.parents = []

    def add_parent(self, neuron, weight):
        self.parents.append([neuron, weight])

    def calculate_temp(self):
        self.temp = sum(neuron[0].activation * neuron[1] for neuron in self.parents)

    def calculate_act(self):
        act = expit(self.temp)
        self.activation = act

    def calculate_parent_sum_activation(self):
        return sum(neuron[0].activation for neuron in self.parents)

    def calculate_parent_sum_weight(self):
        return sum(neuron[1] for neuron in self.parents)
