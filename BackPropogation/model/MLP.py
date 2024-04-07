from typing import List
import numpy as np


class MLP:
    def __init__(self, size: List[int], activate_func):
        self.size = size
        self.activate_func = activate_func
        self.W = []
        self.B = []
        for pre_layer_size, next_layer_size in zip(size[:-1], size[1:]):
            print(pre_layer_size, next_layer_size)
            W = np.random.randn(pre_layer_size, next_layer_size)
            B = np.zeros(next_layer_size)
            self.W.append(W)
            self.B.append(B)

    def print_struct(self):
        for i, (w, b) in enumerate(zip(self.W, self.B)):
            print(f"第{i}层参数 w: {w.shape}, b:{b.shape}")

    def forward(self, x):
        a = x
        for w, b in zip(self.W, self.B):
            z = np.dot(a, x) + b
            a = self.activate_func(z)
        return a

    def backward(self):
        """
        :return:
        """
        pass


model = MLP([784, 30, 10])
model.print_struct()
print(model.W[1])
