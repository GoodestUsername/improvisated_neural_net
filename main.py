import random
import numpy as np
import pickle

from mnist import MNIST
import concurrent.futures
from app.classes.network import Network
from app.classes.neuron import Neuron
import matplotlib.pyplot as plt

from app.utils.normalizers import normalize_layer_weights, normalize_input

possible_outputs = ['0', '1', "2", "3", "4", "5", "6", "7", "8", "9"]


def plot(x, y):
    plt.figure()
    plt.plot(x, y)

    # Add labels and title
    plt.xlabel('X-')
    plt.ylabel('Y-SUM')
    plt.title('SUM')

    # Show the plot


def load_nn():
    with open('state.pkl', 'rb') as f:
        neural_network = pickle.load(f)
    return neural_network


def save_nn(neural_network):
    with open('state.pkl', 'wb') as f:
        pickle.dump(neural_network, f)


def train(neural_network, images, labels):
    completed = 0
    for i in range(5000, 10000):
        neural_network.train(normalize_input(images[i]), labels[i])
        # print("start", start)
        # print("end", end)
        # print("____")
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     futures = []
        #     for i in range(start, end):
        #         futures.append(executor.submit(neural_network.train, normalize_input(images[i]), labels[i]))
        #     for future in concurrent.futures.as_completed(futures):
        #         result = future.result()
        #         completed += 1


def main():
    mndata = MNIST('samples')

    images, labels = mndata.load_training()
    neural_network = load_nn()
    for i in range(100, 110):
        print(mndata.display(images[i]))
        print("should be: " + str(labels[i]))
        neural_network.try_image(normalize_input(images[i]), labels[i])
        print('no work')

    # new nn section
    # neural_network = Network(784, normalize_input(images[0]), labels[0], possible_outputs)
    # neural_network.try_image(normalize_input(images[5000]), labels[5000])
    # neural_network.forward_input()
    # neural_network.get_result()

    # graphs section
    # plot(list(range(len(neural_network.input_layer))), [x.activation for x in neural_network.input_layer])
    # plot(list(range(len(images[5000]))), (images[5000]))
    # plot(list(range(len(images[5000]))), normalize_input(images[5000]))
    # plot(list(range(len(neural_network.hidden_layer))), [x.temp for x in neural_network.hidden_layer])
    # plot(list(range(len(neural_network.hidden_layer))), [x.activation for x in neural_network.hidden_layer])
    # plot(list(range(len(neural_network.output_layer))), [x.temp for key, x in neural_network.output_layer.items()])
    # plot(list(range(len(neural_network.output_layer))), [x.temp for key, x in neural_network.output_layer.items()])
    # neural_network.cleanup()
    # plt.show()

    # training section
    # train(neural_network, images, labels)
    # neural_network.cleanup()
    # save_nn(neural_network)

    # print('yay skynet')



    # index = random.randrange(0, len(images))  # choose an index ;-)
    # print(labels[index])
    # print("hello world")


if __name__ == '__main__':
    main()
