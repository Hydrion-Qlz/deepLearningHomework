import numpy as np


def zero_initialize(input_size, output_size):
    return np.zeros((input_size, output_size))


def random_initialize(input_size, output_size):
    return np.random.randn(input_size, output_size) * 0.01


def Xavier_initialize(input_size, output_size):
    return np.random.randn(input_size, output_size) / np.sqrt(output_size)


def He_initialize(input_size, output_size):
    return np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
