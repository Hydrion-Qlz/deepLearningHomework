import numpy as np
from matplotlib import pyplot as plt

from utils import *

train_images, train_labels = load_mnist_train()
print(train_images.shape, train_labels.shape)
peek_dataset(train_images, train_labels)
#
# import numpy as np
#
# # 架构定义
# input_size = 784
# hidden_size = 30
# output_size = 10
#
# # 初始化权重和偏置参数
# W1 = np.random.randn(input_size, hidden_size)
# b1 = np.zeros(hidden_size)
# W2 = np.random.randn(hidden_size, output_size)
# b2 = np.zeros(output_size)
#
#
# # 前向传播
# def forward(X):
#     # 第一层计算
#     z1 = np.dot(X, W1) + b1
#     a1 = sigmoid(z1)
#
#     # 第二层计算
#     z2 = np.dot(a1, W2) + b2
#     a2 = softmax(z2)
#
#     return a2
#
#
# # 反向传播
# def backward(X, y, output):
#     # 计算输出层的梯度
#     delta2 = output - y
#
#     # 计算隐藏层的梯度
#     delta1 = np.dot(delta2, W2.T) * sigmoid_derivative(np.dot(X, W1) + b1)
#
#     # 计算权重和偏置参数的梯度
#     dW2 = np.dot(a1.T, delta2)
#     db2 = np.sum(delta2, axis=0)
#     dW1 = np.dot(X.T, delta1)
#     db1 = np.sum(delta1, axis=0)
#
#     return dW1, db1, dW2, db2
#
#
# # 参数更新
# def update_parameters(dW1, db1, dW2, db2, learning_rate):
#     W1 -= learning_rate * dW1
#     b1 -= learning_rate * db1
#     W2 -= learning_rate * dW2
#     b2 -= learning_rate * db2
#
#
# # 激活函数和导数定义
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
#
#
# def sigmoid_derivative(x):
#     return sigmoid(x) * (1 - sigmoid(x))
#
#
# def softmax(x):
#     exp_scores = np.exp(x)
#     return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
#
#
# # 训练过程
# num_epochs = 10
# learning_rate = 0.1
#
# for epoch in range(num_epochs):
#     # 在每个epoch开始时，对训练集进行随机打乱
#     np.random.shuffle(train_dataset)
#
#     for X, y in train_dataset:
#         # 前向传播
#         output = forward(X)
#
#         # 反向传播
#         dW1, db1, dW2, db2 = backward(X, y, output)
#
#         # 参数更新
#         update_parameters(dW1, db1, dW2, db2, learning_rate)
#
#     # 在每个epoch结束时，计算训练集上的准确率
#     correct = 0
#     total = 0
#
#     for X, y in train_dataset:
#         output = forward(X)
#         predictions = np.argmax(output, axis=1)
#         correct += np.sum(predictions == y)
#         total += len(y)
#
#     accuracy = correct / total
#     print(f"Epoch {epoch + 1}/{num_epochs} Accuracy: {accuracy}")
