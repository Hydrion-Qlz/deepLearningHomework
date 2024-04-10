def compute_momentum(d_variable, V_d_variable, variable, learning_rate, momentum=0.9):
    # 更新第一层的权重和偏置
    V_d_variable = momentum * V_d_variable + learning_rate * d_variable
    variable -= V_d_variable
    return V_d_variable, variable
