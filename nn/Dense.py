from nn.Variables import Variables
from nn.Activation import Activation
# from Optimizer.Losses import Losses
import numpy as np


class Dense(Variables):

    def __init__(self, unit_size, input_size=None, activation='softmax', weight_params=(-1, 1), bias_params=(0, 1),
                 bias_bool=True, store_vectors=True, ):
        self.store_vectors = store_vectors
        super().__init__(unit_size, input_size, activation, weight_params, bias_params, bias_bool)

        self.A = Activation()
        self.Vectors = {}
        if not bias_bool:
            self.bias_params = (0, 0)

    def feed_forward(self, input_vect):
        x = np.asarray(input_vect)
        h = np.add(np.matmul(x, self.Weight), self.Bias)
        y = self.A.activation_function(h, self.activation)
        if self.store_vectors:
            self.Vectors['x'], self.Vectors['h'], self.Vectors['y'] = x, h, y
        return y

    def gradient(self, error, i, j, training_rate=0.6, momentum_bool=0, gamma=0.9, last_layer=True):
        assert isinstance(error, float) or isinstance(error, int)
        dy_dh = self.A.errors['a'][j]
        dh_w = self.Vectors['x'][i]
        # dh_dx = self.Weight[i][j]
        cost_w, cost_b = error * dy_dh * training_rate, error * dy_dh * training_rate
        if not last_layer:
            cost_w *= dh_w
        delta = momentum_bool*self.W_p_mtx[i][j] + cost_w
        self.Weight[i][j] += delta
        self.errors[i][j] += delta
        self.Bias[j] += momentum_bool*self.B_p_mtx[j] + cost_b
        if momentum_bool:
            self.momentum_handler(i, j, cost_w, cost_b, gamma)
        # return error*dy_dh*dh_dx

    def train(self, loss_vect, training_rate=0.5):
        for j in range(self.unit_size):
            l = loss_vect[j]
            for i in range(self.input_size):
                self.gradient(l, i, j, training_rate)

    def momentum_handler(self, i, j, cost_w, cost_b, gamma):
        self.W_p_mtx[i, j] = gamma*self.W_p_mtx[i, j] + cost_w
        self.B_p_mtx[j] = gamma*self.B_p_mtx[j] + cost_b
        return

    def transformLossTensor(self, loss_tensor):
        assert np.ndim(loss_tensor) == 1
        w = np.linalg.pinv(self.Weight)
        return np.matmul(loss_tensor, w)


def word2vectGenerator(vocab_size):
    lst = np.zeros(vocab_size)
    lst[np.random.randint(0, vocab_size)] = 1
    return lst


def labelGenerator(vocab_size):
    indices = np.random.randint(0, vocab_size, np.random.randint(1, vocab_size-1))
    lst = np.zeros(vocab_size)
    for index in indices:
        lst[index] = 1
    return lst


if __name__ == "__main__":
    i, o = 10, 4
    D = Dense(o, i, activation="sigmoid")
    x = word2vectGenerator(i)
    y = labelGenerator(o)

    for _ in range(10):
        z = D.feed_forward(x)
        l = np.subtract(y, z)
        D.train(l)
        print(z)
    print(y)



