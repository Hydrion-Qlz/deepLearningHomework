import numpy as np

from model import ThreeLayerModel
from utils import random_initialize


class ThreeLayerModel_Adam(ThreeLayerModel):
    def __init__(self, input_size, hidden_size, output_size, param_initialize=random_initialize,
                 beta1=0.9,
                 beta2=0.999, epsilon=1e-8):
        super().__init__(input_size, hidden_size, output_size, param_initialize)

        # 初始化Adam变量
        self.mW1, self.vW1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        self.mb1, self.vb1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        self.mW2, self.vW2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        self.mb2, self.vb2 = np.zeros_like(self.b2), np.zeros_like(self.b2)

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # 初始化时间步

    def update_params(self, dW1, db1, dW2, db2, learning_rate):
        # 增加时间步
        self.t += 1

        # 更新第一层的权重和偏置
        self.mW1 = self.beta1 * self.mW1 + (1 - self.beta1) * dW1
        self.vW1 = self.beta2 * self.vW1 + (1 - self.beta2) * np.square(dW1)
        mW1_corrected = self.mW1 / (1 - np.power(self.beta1, self.t))
        vW1_corrected = self.vW1 / (1 - np.power(self.beta2, self.t))
        self.W1 -= learning_rate * mW1_corrected / (np.sqrt(vW1_corrected) + self.epsilon)

        self.mb1 = self.beta1 * self.mb1 + (1 - self.beta1) * db1
        self.vb1 = self.beta2 * self.vb1 + (1 - self.beta2) * np.square(db1)
        mb1_corrected = self.mb1 / (1 - np.power(self.beta1, self.t))
        vb1_corrected = self.vb1 / (1 - np.power(self.beta2, self.t))
        self.b1 -= learning_rate * mb1_corrected / (np.sqrt(vb1_corrected) + self.epsilon)

        # 更新第二层的权重和偏置
        self.mW2 = self.beta1 * self.mW2 + (1 - self.beta1) * dW2
        self.vW2 = self.beta2 * self.vW2 + (1 - self.beta2) * np.square(dW2)
        mW2_corrected = self.mW2 / (1 - np.power(self.beta1, self.t))
        vW2_corrected = self.vW2 / (1 - np.power(self.beta2, self.t))
        self.W2 -= learning_rate * mW2_corrected / (np.sqrt(vW2_corrected) + self.epsilon)

        self.mb2 = self.beta1 * self.mb2 + (1 - self.beta1) * db2
        self.vb2 = self.beta2 * self.vb2 + (1 - self.beta2) * np.square(db2)
        mb2_corrected = self.mb2 / (1 - np.power(self.beta1, self.t))
        vb2_corrected = self.vb2 / (1 - np.power(self.beta2, self.t))
        self.b2 -= learning_rate * mb2_corrected / (np.sqrt(vb2_corrected) + self.epsilon)

    def save_parameter(self, file_path, **kwargs):
        super().save_parameter(file_path, beta1=self.beta1, beta2=self.beta2,
                               epsiilon=self.epsilon, t=self.t,
                               mW1=self.mW1, vW1=self.vW1,
                               mb1=self.mb1, vb1=self.vb1,
                               mW2=self.mW2, vW2=self.vW2,
                               mb2=self.mb2, vb2=self.vb2)

    def load_parameter(self, file_path):
        params = super().load_parameter(file_path)
        self.mW1, self.vW1 = params['mW1'], params['vW1']
        self.mb1, self.vb1 = params['mb1'], params['vb1']
        self.mW2, self.vW2 = params['mW2'], params['vW2']
        self.mb2, self.vb2 = params['mb2'], params['vb1']

        self.beta1 = params['beta1']
        self.beta2 = params['beta2']
        self.epsilon = params['epsilon']
        self.t = params['t']  # 初始化时间步
