from math import e
import numpy as np


class ConvActivations:
    def __init__(self, alpha=0.2, func='leaky_relu'):
        self.alpha = alpha
        self.func = func

    @staticmethod
    def relu(number):
        return max(0, number)

    def leaky_relu(self, number):
        return self.alpha*number if number < 0 else number

    def linear(self, num):
        return num

    @staticmethod
    def sigmoid(num):
        return 1/(1+(e**(-num)))

    def tanh(self, num):
        return np.tanh(num)

    def softplus(self, num):
        return np.log(1 + (e**num))

    ''' derivatives '''
    @staticmethod
    def linear_derivative(*args):
        return 1

    def sigmoid_derivative(self, num):
        return self.sigmoid(num)*(1-self.sigmoid(num))

    def tanh_derivative(self, num):
        return 1 - np.square(np.tanh(num))

    def softplus_derivative(self, num):
        self.sigmoid(num)

    @staticmethod
    def relu_derivative(number):
        return 0 if number < 0 else 1

    def leaky_relu_derivative(self, number):
        return self.alpha if number < 0 else 1

    def activation_function(self, number):
        dct = {'linear': self.linear, 'sigmoid': self.sigmoid, 'tanh': self.tanh, 'relu': self.relu,
               'leaky_relu': self.leaky_relu, 'softplus': self.softplus}
        try:
            return dct[self.func](number), self.activation_function_derivative(number)
        except KeyError:
            return number, 1

    def activation_function_derivative(self, number):
        dct = {'linear': self.linear_derivative, 'sigmoid': self.sigmoid_derivative, 'tanh': self.tanh_derivative,
               'relu': self.relu_derivative, 'leaky_relu': self.leaky_relu_derivative,
               'softplus': self.softplus_derivative}
        try:
            return dct[self.func](number)
        except KeyError:
            return 1

