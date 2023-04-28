import numpy as np
from scipy.special import expit


def sigmoid(x):
    sigmoid_x = expit(x)

    if x >= 0:
        return 1 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))


def derivative_sigmoid(x):
    return expit(x) * (1 - expit(x))
