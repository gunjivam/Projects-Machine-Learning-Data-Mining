from abc import ABC, abstractmethod
from nn.Activation import Activation
import numpy as np


class AbstractRNN(ABC):

    def __init__(self, input_size, hidden_size,  output_activation="softmax", hidden_activation='tanh',
                 weight_param=(0, 5), bias_params=(-1, 1), bias_bool=True, training_iterations=3):
        self.Vectors = {}
        self.timestamp = 1
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        self.weight_params = weight_param
        self.bias_params = bias_params
        self.bias_bool = bias_bool
        self.A = Activation()
        self.Vectors['h0'] = np.zeros(hidden_size)
        self.iterations = training_iterations

    @abstractmethod
    def initiate_weights(self):
        pass

    @abstractmethod
    def feed_forward_one_vect(self, input_vect):
        pass

    @abstractmethod
    def train(self, error):
        pass

    def get_output_abs(self, vect):
        return self.Vectors[vect+str(self.timestamp-1)]

    def feed_forward(self, input_vects):
        assert np.asarray(input_vects).ndim == 2
        self.timestamp = 1
        for vector in input_vects:
            self.feed_forward_one_vect(vector)
