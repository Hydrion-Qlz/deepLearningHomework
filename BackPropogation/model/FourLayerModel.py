import numpy as np

from utils import sigmoid, softmax, sigmoid_derivative


class FourLayerModel:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size1) * 0.01
        self.b1 = np.zeros((1, hidden_size1))
        self.W2 = np.random.randn(hidden_size1, hidden_size2) * 0.01
        self.b2 = np.zeros((1, hidden_size2))
        self.W3 = np.random.randn(hidden_size2, output_size) * 0.01
        self.b3 = np.zeros((1, output_size))

    # Forward propagation
    def forward(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = sigmoid(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = sigmoid(Z2)
        Z3 = np.dot(A2, self.W3) + self.b3
        A3 = softmax(Z3)
        return Z1, A1, Z2, A2, Z3, A3

    # Backward propagation
    def backward(self, X, Y, Z1, A1, Z2, A2, Z3, A3):
        m = X.shape[0]

        dZ3 = A3 - Y
        dW3 = np.dot(A2.T, dZ3) / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m

        dA2 = np.dot(dZ3, self.W3.T)
        dZ2 = dA2 * sigmoid_derivative(A2)
        dW2 = np.dot(A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * sigmoid_derivative(A1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        return dW1, db1, dW2, db2, dW3, db3

    # Update parameters
    def update_params(self, dW1, db1, dW2, db2, dW3, db3, learning_rate):
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3

    # Prediction
    def predict(self, X):
        _, _, _, _, _, A3 = self.forward(X)
        return np.argmax(A3, axis=1)

    def normalization_loss(self):
        return 0

    def save_parameter(self, file_path, **kwargs):
        np.savez(file_path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, W3=self.W3, b3=self.b3, **kwargs)

    def load_parameter(self, file_path):
        parameters = np.load(file_path)
        self.W1 = parameters['W1']
        self.b1 = parameters['b1']
        self.W2 = parameters['W2']
        self.b2 = parameters['b2']
        self.W3 = parameters['W3']
        self.b3 = parameters['b3']
        return parameters
