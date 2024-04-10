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
        images = images.reshape(num_images, -1)
        return images


def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = f.read()
        magic_number = int.from_bytes(data[0:4], 'big')
        num_items = int.from_bytes(data[4:8], 'big')

        labels = np.frombuffer(data, dtype=np.uint8, offset=8)
        num_classes = len(np.unique(labels))

        # One-Hot 编码
        one_hot_labels = np.eye(num_classes)[labels]
        return one_hot_labels


def peek_dataset(images, labels):
    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(30):
        image = np.reshape(images[i], [28, 28])
        ax = fig.add_subplot(6, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(image, cmap=plt.cm.binary, interpolation='nearest')
        ax.text(0, 7, str(labels[i]))
    plt.show()


def plot_result_figure(
        train_loss_lst,
        test_loss_lst,
        train_accuracy_lst,
        test_accuracy_lst,
        figure_title,
        save_path="result.png"):
    epochs = range(1, len(train_loss_lst) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), sharex=True)

    ax1.plot(epochs, train_loss_lst, label='Training Loss', color='tab:blue')
    ax1.plot(epochs, test_loss_lst, label='Testing Loss', color='tab:orange')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()

    ax2.plot(epochs, train_accuracy_lst, label='Training Accuracy', color='tab:green')
    ax2.plot(epochs, test_accuracy_lst, label='Testing Accuracy', color='tab:red')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()

    fig.suptitle(figure_title, fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.show()
    fig.savefig(save_path)


# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)


# Softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


# Loss function - Categorical Crossentropy
def CrossEntropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / m
    return loss


def compute_loss(images, labels, model):
    _, _, _, A2_full = model.forward(images)
    return CrossEntropy_loss(labels, A2_full)


def compute_accuracy(images, labels, model):
    predictions = model.predict(images)
    return np.mean(np.argmax(labels, axis=1) == predictions)


def print_epoch_result(epoch, test_accuracy_lst, test_loss_lst, train_accuracy_lst, train_loss_lst):
    print(f"epoch {epoch + 1}")
    print(f"Training Loss: {train_loss_lst[-1]}")
    print(f"Testing Loss: {test_loss_lst[-1]}")
    print(f"Training Accuracy: {train_accuracy_lst[-1]}")
    print(f"Testing Accuracy: {test_accuracy_lst[-1]}\n")
