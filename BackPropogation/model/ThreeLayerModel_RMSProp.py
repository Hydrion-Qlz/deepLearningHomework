import numpy as np

from model import ThreeLayerModel
from utils import random_initialize


class ThreeLayerModel_RMSProp(ThreeLayerModel):
    def __init__(self, input_size, hidden_size, output_size, param_initialize=random_initialize,
                 decay_rate=0.99,
                 epsilon=1e-8):
        super().__init__(input_size, hidden_size, output_size, param_initialize)

        # 初始化RMSProp变量
        self.sW1, self.sb1 = np.zeros_like(self.W1), np.zeros_like(self.b1)
        self.sW2, self.sb2 = np.zeros_like(self.W2), np.zeros_like(self.b2)

        self.decay_rate = decay_rate
        self.epsilon = epsilon

    def update_params(self, dW1, db1, dW2, db2, learning_rate):
        # 更新第一层的权重和偏置的RMSProp
        self.sW1 = self.decay_rate * self.sW1 + (1 - self.decay_rate) * np.square(dW1)
        self.W1 -= learning_rate * dW1 / (np.sqrt(self.sW1) + self.epsilon)

        self.sb1 = self.decay_rate * self.sb1 + (1 - self.decay_rate) * np.square(db1)
        self.b1 -= learning_rate * db1 / (np.sqrt(self.sb1) + self.epsilon)

        # 更新第二层的权重和偏置的RMSProp
        self.sW2 = self.decay_rate * self.sW2 + (1 - self.decay_rate) * np.square(dW2)
        self.W2 -= learning_rate * dW2 / (np.sqrt(self.sW2) + self.epsilon)

        self.sb2 = self.decay_rate * self.sb2 + (1 - self.decay_rate) * np.square(db2)
        self.b2 -= learning_rate * db2 / (np.sqrt(self.sb2) + self.epsilon)

    def save_parameter(self, file_path, **kwargs):
        super().save_parameter(file_path, decay_rate=self.decay_rate, epsilon=self.epsilon,
                               sW1=self.sW1, sb1=self.sb1,
                               sW2=self.sW2, sb2=self.sb2)

    def load_parameter(self, file_path):
        params = super().load_parameter(file_path)

        self.decay_rate = params["decay_rate"]
        self.epsilon = params["epsilon"]
        self.sW1, self.sb1 = params["sW1"], params["sb1"]
        self.sW2, self.sb2 = params["sW2"], params["sb2"]
