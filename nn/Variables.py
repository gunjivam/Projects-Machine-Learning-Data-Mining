import numpy as np


class Variables:
    gamma1, gamma2 = 1, 0
    Weight, Bias = [], []
    W_p_mtx, B_p_mtx = [], []
    errors = []

    def __init__(self, unit_size, input_size, activation, weight_params, bias_params, bias_bool):
        self.input_size = input_size
        if not bias_bool:
            bias_params = (0, 0)
        self.unit_size, self.activation, self.weight_params, self.bias_params, self.biases_bool = \
            unit_size, activation, weight_params, bias_params, bias_bool

        if input_size is not None:
            self.initialize()

    def initialize(self):
        self.Weight = np.random.uniform(self.weight_params[0], self.weight_params[1],
                                        size=(self.input_size, self.unit_size))
        self.Bias = np.random.uniform(self.weight_params[0], self.weight_params[1], size=self.unit_size)
        self.W_p_mtx = np.zeros((self.input_size, self.unit_size))
        self.B_p_mtx = np.zeros(self.unit_size)
        self.errors = np.zeros((self.input_size, self.unit_size))

