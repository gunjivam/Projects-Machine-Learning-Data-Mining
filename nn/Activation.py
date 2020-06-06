import numpy as np
from math import e


class Activation(object):

    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.errors = {}
        self.dct = {'linear': self.linear, 'sigmoid': self.sigmoid_wrapper, 'tanh': self.tanh, 'relu': self.relu,
               'leaky_relu': self.leaky_relu, 'softplus': self.softplus, 'softmax': self.softmax_wrapper}

    def linear(self, vector, vect_name):
        self.linear_derivative(vector, vect_name)
        return vector

    @staticmethod
    def sigmoid(vector):
        func = lambda x: 1/(1+(e**(-x)))
        return np.asarray(list(map(func, vector)))

    def sigmoid_wrapper(self, vector, vect_name):
        v = self.sigmoid(vector)
        self.sigmoid_derivative(vector, vect_name)
        return v

    def tanh(self, vector, vect_name):
        self.tanh_derivative(vector, vect_name)
        return np.tanh(vector)

    def relu(self, vector, vect_name):
        self.relu_derivative(vector, vect_name)
        return np.asarray(map(lambda x: max(0, x), vector))

    def leaky_relu(self, vector, vect_name):
        self.leaky_relu_derivative(vector, vect_name)
        return np.asarray(map(lambda x: self.alpha*x if x < 0 else x, vector))

    def softplus(self, vector, vect_name):
        self.softplus_derivative(vector, vect_name)
        return np.log(map(lambda x: 1 + (e**x), vector))

    @staticmethod
    def softmax(vector):
        return np.divide(np.exp(vector), np.sum(np.exp(vector)))

    def softmax_wrapper(self, vector, vect_name):
        v = self.softmax(vector)
        self.softmax_derivative(vector, vect_name)
        return v

    ''' derivatives '''
    def linear_derivative(self, vector, vect_name):
        self.errors[vect_name] = np.array([1 for i in range(len(vector))])

    def sigmoid_derivative(self, vector, vect_name):
        self.errors[vect_name] = self.sigmoid(vector)*(1-self.sigmoid(vector))

    def tanh_derivative(self, vector, vect_name):
        self.errors[vect_name] = 1 - np.square(np.tanh(vector))

    def relu_derivative(self, vector, vect_name):
        self.errors[vect_name] = np.asarray(map(lambda x: 0 if x < 0 else 1, vector))

    def leaky_relu_derivative(self, vector, vect_name):
        self.errors[vect_name] = np.asarray(map(lambda x: self.alpha if x < 0 else 1, vector))

    def softplus_derivative(self, vector, vect_name):
        self.errors[vect_name] = self.sigmoid(vector)

    def softmax_derivative(self, vector, vect_name):
        self.errors[vect_name] = np.zeros(len(vector))
        a = self.softmax(vector)
        for i in range(len(vector)):
            self.errors[vect_name][i] = a[i]*(1 - a[i])

    def activation_function(self, vector, act_function, vect_name="a"):
        if act_function is None:
            self.errors[vect_name] = np.asarray([1 for i in range(len(vector))])
            return vector
        else:
            return self.dct[act_function](vector, vect_name)

