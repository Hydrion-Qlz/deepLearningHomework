import time

import numpy as np

from model.MLP import MLP, sigmoid, sigmoid_derivative, cross_entropy_loss, relu, relu_derivative
from utils import *

train_images, train_labels = load_mnist_train()
test_images, test_labels = load_mnist_test()

size = [784, 256, 10]
learning_rate = 0.1

# model = MLP(size, learning_rate, sigmoid, sigmoid_derivative)
model = MLP(size, learning_rate, relu, relu_derivative)
num_epochs = 10
batch_size = 500


def get_train_data(train_images, train_labels, batch_size):
    num_samples = train_images.shape[0]
    num_batches = num_samples // batch_size
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size

        batch_images = train_images[start_idx:end_idx]
        batch_labels = train_labels[start_idx:end_idx]
        yield batch_idx, batch_images, batch_labels


for epoch in range(num_epochs):
    start_time = time.time()
    print(f"Epoch: {epoch + 1} starts.")
    train_loss = []
    for _, X, y in get_train_data(train_images, train_labels, batch_size):
        output = model.forward(X)

        loss = cross_entropy_loss(output, y)
        train_loss.append(loss)
        dW, dB = model.backward(X, y, output)

        model.update_parameter(dW, dB)
    train_end_time = time.time()

    # 计算训练集上的正确率
    correct = 0
    total = 0

    for X, y in zip(train_images, train_labels):
        output = model.forward(X)
        prediction = np.argmax(output, axis=1)
        correct += np.sum(prediction == y)
        total += len(y)

    accuracy = correct / total
    pred_end_time = time.time()
    print(f"Epoch {epoch + 1}/{num_epochs} Accuracy: {accuracy}")
    print(f"Train time: {train_end_time - start_time}, Test time: {pred_end_time - train_end_time}")
