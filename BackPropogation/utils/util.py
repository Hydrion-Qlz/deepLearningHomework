import gzip
import os
import struct

import numpy as np
from matplotlib import pyplot as plt


def load_mnist_train():
    images = load_mnist_images("dataset/train-images-idx3-ubyte.gz")
    labels = load_mnist_labels("dataset/train-labels-idx1-ubyte.gz")
    return images, labels


def load_mnist_test():
    images = load_mnist_images("dataset/t10k-images-idx3-ubyte.gz")
    labels = load_mnist_labels("dataset/t10k-labels-idx1-ubyte.gz")
    return images, labels


def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = f.read()
        magic_number = int.from_bytes(data[0:4], 'big')
        num_images = int.from_bytes(data[4:8], 'big')
        num_rows = int.from_bytes(data[8:12], 'big')
        num_cols = int.from_bytes(data[12:16], 'big')

        images = np.frombuffer(data, dtype=np.uint8, offset=16)
        images = images.reshape(num_images, num_rows, num_cols)
        return images


def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = f.read()
        magic_number = int.from_bytes(data[0:4], 'big')
        num_items = int.from_bytes(data[4:8], 'big')

        labels = np.frombuffer(data, dtype=np.uint8, offset=8)
        return labels


def peek_dataset(images, labels):
    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(30):
        image = np.reshape(images[i], [28, 28])
        ax = fig.add_subplot(6, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(image, cmap=plt.cm.binary, interpolation='nearest')
        ax.text(0, 7, str(labels[i]))
    plt.show()
