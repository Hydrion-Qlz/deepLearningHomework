import numpy as np

from model import ThreeLayerModel
from utils import random_initialize, sigmoid_derivative


class ThreeLayerModel_L2_Normalization(ThreeLayerModel):
    def __init__(self, input_size, hidden_size, output_size, param_initialize=random_initialize, lamda=1e-3):
        # Initialize weights and biases
        super().__init__(input_size, hidden_size, output_size, param_initialize)
        self.lamda = lamda

    def backward(self, X, Y, Z1, A1, Z2, A2):
        m = X.shape[0]

        dZ2 = A2 - Y
        dW2 = np.dot(A1.T, dZ2) / m + self.lamda * self.W2
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * sigmoid_derivative(A1)
        dW1 = np.dot(X.T, dZ1) / m + self.lamda * self.W1
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        return dW1, db1, dW2, db2

    def normalization_loss(self):
        return self.lamda * 0.5 * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))
