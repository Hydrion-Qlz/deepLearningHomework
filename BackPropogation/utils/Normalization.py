import numpy as np


def l1_regularization(weights, lamda):
    return lamda * np.sum(np.abs(weights))


def l2_regularization(weights, lamda):
    return lamda * 0.5 * np.sum(np.square(weights))
