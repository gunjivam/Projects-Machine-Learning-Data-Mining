from nn.Dense import Dense
import numpy as np
from nn.rnn.AbstractRNN import AbstractRNN


class RNNCell(AbstractRNN):
    """Vectors: all hidden and cell state vectors"""

    def __init__(self, hidden_size, input_size=None, training_iterations=5, hidden_activation='softmax', weight_param=(-1, 1)
                 , bias_params=(-1, 1), bias_bool=True, fp=''):
        super().__init__(input_size, hidden_size,  None, hidden_activation, (0, 5), (-1, 1), True, training_iterations)

        self.I = Dense(hidden_size, input_size, None, weight_param, bias_params, False)
        self.H = Dense(hidden_size, hidden_size, None, weight_param, bias_params, bias_bool)

        self.dh_dW1 = np.zeros(shape=(input_size, hidden_size))
        self.dh_dWh = np.zeros(shape=(hidden_size, hidden_size))
        self.dh_db = np.zeros(hidden_size)

        if fp == '':
            self.initiate_weights()

    def initiate_weights(self):
        self.I.initialize(), self.H.initialize()

    def feed_forward_one_vect(self, input_vect):
        h1 = self.I.feed_forward(input_vect)
        h2 = self.H.feed_forward(self.Vectors['h'+str(self.timestamp-1)])
        h = np.add(h1, h2)
        y = self.A.activation_function(h, self.hidden_activation, "a"+str(self.timestamp))

        self.Vectors['x' + str(self.timestamp)] = input_vect
        self.Vectors['h'+str(self.timestamp)] = y
        self.timestamp += 1
        return y

    def train(self, error_vect):
        timestamp = 1
        while timestamp < self.timestamp:
            max_timestamp = min(timestamp + self.iterations, self.timestamp)
            while timestamp < max_timestamp:
                for j in range(self.hidden_size):
                    h_prv = self.Vectors['h' + str(timestamp - 1)][j]
                    activation_error = self.A.errors["a" + str(timestamp)][j]
                    error = error_vect[j]
                    self.H.Bias[j] += error * self.dB1(activation_error, self.H.Weight[j][j], self.dh_db[j], j)
                    for i in range(self.input_size):
                        x = self.Vectors['x' + str(timestamp)][i]
                        self.I.Weight[i][j] += error * self.dW1(activation_error, x, self.H.Weight[j][j],
                                                                self.dh_dW1[i][j], i, j)

                    for j2 in range(self.hidden_size):
                        self.H.Weight[j2][j] += error * self.dWh(activation_error, h_prv, self.H.Weight[j2][j],
                                                                 self.dh_dWh[j2][j], j2, j)

                timestamp += 1
            self.reset()

    def dW1(self, ae, x, wh, dh_prv, i, j):
        er = (x + wh*dh_prv)*ae
        self.dh_dW1[i][j] = er
        return er

    def dWh(self, ae, h_prv, wh, dh_prv, i, j):
        er = ae*(h_prv + wh*dh_prv)
        self.dh_dWh[i][j] = er
        return er

    def dB1(self, ae, wh, dh_prv, j):
        er = ae*(wh*dh_prv + 1)
        self.dh_db[j] = er
        return er

    def dNext(self, error, ae, wh, dh_prv, i, j):
        er = (self.I.Weight[i][j] + wh * dh_prv) * ae * error
        return er

    def reset(self):
        self.dh_dW1 = np.zeros(shape=(self.I.input_size, self.hidden_size))
        self.dh_dWh = np.zeros(shape=(self.hidden_size, self.hidden_size))
        self.dh_db = np.zeros(self.hidden_size)

    def get_output(self):
        return self.Vectors['h'+str(self.timestamp-1)]


def test():
    s, t = 0, 0
    for _ in range(100):
        i, o = 20, 10
        R = RNNCell(o, i, weight_param=(0, 1), bias_params=(0, 1), hidden_activation='softmax')
        vects = [np.random.random(size=i) for j in range(i)]

        y = np.zeros(o)
        y[1] = 1

        # R.feed_forward(vects[0])

        for vect in vects:
            R.feed_forward(vect)

        for i in range(20):
            error_vect = np.subtract(y, R.get_output())
            R.train(np.multiply(error_vect, 0.06))

            for vect in vects:
                R.feed_forward(vect)

        t += 1
        if np.argmax(R.get_output()) == 1:
            s += 1

    return s / t


if __name__ == '__main__':
    print(test())
    # s, t = 0, 1
    #
    # i, o = 5, 5
    # R = RNNCell(i, o)
    # vects = [np.random.random(size=i) for j in range(4)]
    #
    # y = np.zeros(o)
    # y[1] = 1
    #
    # for vect in vects:
    #     R.feed_forward(vect)
    # print(R.Vectors.keys())
    # print(R.timestamp)
    # for _ in range(20):
    #     error_vect = np.subtract(y, R.get_output())
    #     R.gradient(np.multiply(error_vect, 0.05))
    #     for vect in vects:
    #         R.feed_forward(vect)
    #     # print(R.get_output())
    #
    # print(R.Vectors.keys())
    # print(R.timestamp)
    # print(R.dh_db)
    # print(R.I.Vectors.keys())
