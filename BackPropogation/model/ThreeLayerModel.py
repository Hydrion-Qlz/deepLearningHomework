import numpy as np

from utils import random_initialize, sigmoid, softmax, sigmoid_derivative


class ThreeLayerModel:
    def __init__(self, input_size, hidden_size, output_size, param_initialize=random_initialize):
        # Initialize weights and biases
        self.W1 = param_initialize(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = param_initialize(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    # Forward propagation
    def forward(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = sigmoid(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = softmax(Z2)
        return Z1, A1, Z2, A2

    # Backward propagation
    def backward(self, X, Y, Z1, A1, Z2, A2):
        m = X.shape[0]

        dZ2 = A2 - Y
        dW2 = np.dot(A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * sigmoid_derivative(A1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        return dW1, db1, dW2, db2

    # Update parameters
    def update_params(self, dW1, db1, dW2, db2, learning_rate):
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    # Prediction
    def predict(self, X):
        _, _, _, A2 = self.forward(X)
        return np.argmax(A2, axis=1)

    def normalization_loss(self):
        return 0

    def save_parameter(self, file_path, **kwargs):
        np.savez(file_path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, **kwargs)

    def load_parameter(self, file_path):
        parameters = np.load(file_path)
        self.W1 = parameters['W1']
        self.b1 = parameters['b1']
        self.W2 = parameters['W2']
        self.b2 = parameters['b2']
        return parameters
