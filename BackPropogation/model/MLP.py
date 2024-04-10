from typing import List
import numpy as np


class MLP:
    def __init__(self,
                 size: List[int],
                 learning_rate=1e-2,
                 activate_func=None,
                 activate_func_derivative=None):
        self.size = size
        self.learning_rate = learning_rate
        self.activate_func = activate_func
        self.activate_func_derivative = activate_func_derivative
        self.W = []
        self.B = []
        self.Z = []
        for pre_layer_size, next_layer_size in zip(size[:-1], size[1:]):
            # print(pre_layer_size, next_layer_size)
            W = np.random.randn(pre_layer_size, next_layer_size) * 0.01
            B = np.zeros(next_layer_size)
            self.W.append(W)
            self.B.append(B)

    def print_struct(self):
        for i, (w, b) in enumerate(zip(self.W, self.B)):
            print(f"第{i}层参数 w: {w.shape}, b:{b.shape}")

    def forward(self, X):
        a = X
        self.Z = [X]
        idx = 0
        for w, b in zip(self.W, self.B):
            z = np.dot(a, w) + b
            self.Z.append(z)
            if idx < (len(self.W) - 1):
                idx += 1
                a = self.activate_func(z)
            else:
                a = softmax(z)
        return a

    def backward(self, X, y, output):
        """
        :return:
        """
        m = X.shape[0]  # 样本数量

        # 初始化梯度
        dW = [np.zeros_like(w) for w in self.W]
        dB = [np.zeros_like(b) for b in self.B]

        # 计算输出层的梯度
        delta = output - y  # (5, 10)

        # 反向传播梯度
        for layer in range(len(self.W) - 1, -1, -1):
            # 计算当前层的梯度
            # X: (5, 784)
            dW[layer] = np.dot(self.Z[layer].T, delta) / m
            dB[layer] = np.mean(delta, axis=0)
            if layer > 0:
                # 更新下一层的误差
                delta = np.dot(delta, self.W[layer].T) * self.activate_func_derivative(self.Z[layer])

        return dW, dB

    def update_parameter(self, dW, dB):
        for i in range(len(self.W)):
            self.W[i] -= self.learning_rate * dW[i]
            self.B[i] -= self.learning_rate * dB[i]


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def sigmoid_derivative(X):
    y = sigmoid(X)
    return y / (1 - y)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def leaky_relu(x, alpha):
    return np.where(x >= 0, x, alpha * x)


def leaky_relu_derivative(x, alpha):
    return np.where(x >= 0, 1, alpha)


def softmax(x):
    exp_values = np.exp(x)
    probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    return probs


def cross_entropy_loss(y_pred, y_true):
    epsilon = 1e-7  # 添加一个小的常数，避免log(0)的情况
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # 对预测概率进行裁剪，避免取对数时出现无穷大
    loss = -np.sum(y_true * np.log(y_pred))  # 计算交叉熵损失
    return loss

#
# model = MLP([784, 30, 10], sigmoid, sigmoid_derivative)
# model.print_struct()
# print(model.W[1])
