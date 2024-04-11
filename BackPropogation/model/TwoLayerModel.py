import numpy as np

from utils import softmax


class TwoLayerModel:
    def __init__(self, input_size, output_size):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, output_size) * 0.01
        self.b1 = np.zeros((1, output_size))

    # Forward propagation
    def forward(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = softmax(Z1)
        return Z1, A1

    # Backward propagation
    def backward(self, X, Y, Z1, A1):
        m = X.shape[0]

        dZ1 = A1 - Y
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        return dW1, db1

    # Update parameters
    def update_params(self, dW1, db1, learning_rate):
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    # Prediction
    def predict(self, X):
        _, A1 = self.forward(X)
        return np.argmax(A1, axis=1)

    def normalization_loss(self):
        return 0

    def save_parameter(self, file_path, **kwargs):
        np.savez(file_path, W1=self.W1, b1=self.b1, **kwargs)

    def load_parameter(self, file_path):
        parameters = np.load(file_path)
        self.W1 = parameters['W1']
        self.b1 = parameters['b1']
        return parameters
