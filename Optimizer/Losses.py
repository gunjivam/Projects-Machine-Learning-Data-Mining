import numpy as np


class Losses(object):
    losses = None
    t, gamma = 1, 0

    def quadratic_cost(self, predicted, expected):
        self.quadratic_cost_gradient(predicted, expected)
        return 0.5*np.sum(np.square(predicted-expected))

    def cross_entropy(self, predicted, expected):
        assert isinstance(predicted, np.ndarray) and isinstance(expected, np.ndarray)
        self.cross_entropy_gradient(predicted, expected)
        return - np.sum(np.multiply(expected, np.log(predicted)))

    def exponential_cost(self, predicted, expected):
        return self.t*np.exp((1/self.t)*np.sum(predicted, expected))

    def exponential_cost_wrapper(self, predicted, expected):
        self.exponential_cost_gradient(predicted, expected)
        return self.exponential_cost(predicted, expected)

    def quadratic_cost_gradient(self, predicted, expected):
        self.losses = expected - predicted

    def cross_entropy_gradient(self, predicted, expected):
        self.losses = list(map(lambda p, e: e/p, predicted, expected))
        # self.losses = (predicted - expected)/(np.multiply((1 - predicted), predicted))

    def exponential_cost_gradient(self, predicted, expected):
        self.losses = (2/self.t)*np.multiply((predicted - expected), self.exponential_cost(predicted, expected))

    def loss_function(self, loss_name, expected, predicted, *args):
        dct = {'quadratic': self.quadratic_cost, 'cross_entropy': self.cross_entropy,
               'exponential': self.exponential_cost}
        try:
            self.t = args[0]
            self.gamma = args[1]
        except IndexError:
            pass
        cost = np.sum(dct[loss_name](predicted, expected))
        return self.losses, cost

#
# def test():
#     m = 5
#     n = 2
#     a = np.random.random((m, n))
#     x = [i for i in range(1, m+1)]
#     y = np.matmul(x, a)
#     if m > n:
#         print(np.matmul(np.linalg.pinv(a), y))
#     elif m < n:
#         print(np.matmul(y, np.linalg.pinv(a)))
#     else:
#         if np.linalg.det(a) != 0:
#             print(np.matmul(y, np.linalg.inv(a)))
#
#
# for _ in range(20):
#     test()

