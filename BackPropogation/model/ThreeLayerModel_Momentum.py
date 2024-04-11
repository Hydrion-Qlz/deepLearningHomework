import numpy as np

from model import ThreeLayerModel
from utils import random_initialize


class ThreeLayerModel_Momentum(ThreeLayerModel):
    def __init__(self, input_size, hidden_size, output_size, param_initialize=random_initialize, momentum=0.9):
        super().__init__(input_size, hidden_size, output_size, param_initialize)
        self.momentum = momentum

        # 初始动量
        self.VdW1 = np.zeros_like(self.W1)
        self.Vdb1 = np.zeros_like(self.b1)
        self.VdW2 = np.zeros_like(self.W2)
        self.Vdb2 = np.zeros_like(self.b2)

    def update_params(self, dW1, db1, dW2, db2, learning_rate, ):
        self.VdW1 = self.momentum * self.VdW1 + learning_rate * dW1
        self.W1 -= self.VdW1
        self.Vdb1 = self.momentum * self.Vdb1 + learning_rate * db1
        self.b1 -= self.Vdb1

        self.VdW2 = self.momentum * self.VdW2 + learning_rate * dW2
        self.W2 -= self.VdW2
        self.Vdb2 = self.momentum * self.Vdb2 + learning_rate * db2
        self.b2 -= self.Vdb2

    def save_parameter(self, file_path, **kwargs):
        super().save_parameter(file_path, momentum=self.momentum,
                               VdW1=self.VdW1,
                               Vdb1=self.Vdb1,
                               VdW2=self.VdW2,
                               Vdb2=self.Vdb2)

    def load_parameter(self, file_path):
        params = super().load_parameter(file_path)

        self.VdW1 = params["VdW1"]
        self.Vdb1 = params["Vdb1"]
        self.VdW2 = params["VdW2"]
        self.Vdb2 = params["Vdb2"]
