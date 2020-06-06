from nn.rnn.AbstractRNN import AbstractRNN
from nn.Dense import Dense
import numpy as np


class LSTMCell(AbstractRNN):
    # g, i, f, o

    def __init__(self, hidden_size, input_size=None, gate_activation="tanh", hidden_activation='tanh',
                 weight_param=(-1, 1), bias_params=(-1, 1), bias_bool=True, fp='', training_iterations=5):
        super().__init__(input_size, hidden_size, gate_activation, hidden_activation, weight_param,
                         bias_params, bias_bool, training_iterations)

        self.Vectors['s0'] = np.zeros(hidden_size)

        self.IG = Dense(hidden_size, input_size, None, weight_param, bias_params, bias_bool, False)
        self.IH = Dense(hidden_size, hidden_size, None, weight_param, bias_params, False, False)

        self.FG = Dense(hidden_size, input_size, None, weight_param, bias_params, bias_bool, False)
        self.FH = Dense(hidden_size, hidden_size, None, weight_param, bias_params, False, False)

        self.GG = Dense(hidden_size, input_size, None, weight_param, bias_params, bias_bool, False)
        self.GH = Dense(hidden_size, hidden_size, None, weight_param, bias_params, False, False)

        self.OG = Dense(hidden_size, input_size, None, weight_param, bias_params, bias_bool, False)
        self.OH = Dense(hidden_size, hidden_size, None, weight_param, bias_params, False, False)

        self.dhp_dWi = np.zeros(shape=(input_size, hidden_size))
        self.dhp_dWf = np.zeros(shape=(input_size, hidden_size))
        self.dhp_dWg = np.zeros(shape=(input_size, hidden_size))
        self.dhp_dWo = np.zeros(shape=(input_size, hidden_size))
        self.dhp_dUi = np.zeros(shape=(hidden_size, hidden_size))
        self.dhp_dUf = np.zeros(shape=(hidden_size, hidden_size))
        self.dhp_dUg = np.zeros(shape=(hidden_size, hidden_size))
        self.dhp_dUo = np.zeros(shape=(hidden_size, hidden_size))
        self.dhp_dBi = np.zeros(hidden_size)
        self.dhp_dBg = np.zeros(hidden_size)
        self.dhp_dBf = np.zeros(hidden_size)
        self.dhp_dBo = np.zeros(hidden_size)

        self.dc_dWi = np.zeros(shape=(input_size, hidden_size))
        self.dc_dUi = np.zeros(shape=(hidden_size, hidden_size))
        self.dc_dWf = np.zeros(shape=(input_size, hidden_size))
        self.dc_dUf = np.zeros(shape=(hidden_size, hidden_size))
        self.dc_dWg = np.zeros(shape=(input_size, hidden_size))
        self.dc_dUg = np.zeros(shape=(hidden_size, hidden_size))
        self.dc_dBi = np.zeros(hidden_size)
        self.dc_dBf = np.zeros(hidden_size)
        self.dc_dBg = np.zeros(hidden_size)

        if fp == '':
            self.initiate_weights()

    def initiate_weights(self):
        self.IG.initialize(), self.IH.initialize(), self.GG.initialize(), self.GH.initialize()
        self.FG.initialize(), self.FH.initialize(), self.OG.initialize(), self.OH.initialize()

    def feed_forward_one_vect(self, input_vect):
        h_prev = self.Vectors['h'+str(self.timestamp-1)]
        s_prev = self.Vectors['s'+str(self.timestamp-1)]
        g = self.A.activation_function(np.add(self.GG.feed_forward(input_vect), self.GH.feed_forward(h_prev)),
                                       self.hidden_activation, "g"+str(self.timestamp))
        i = self.A.activation_function(np.add(self.IG.feed_forward(input_vect), self.IH.feed_forward(h_prev)),
                                       self.output_activation, 'i'+str(self.timestamp))
        f = self.A.activation_function(np.add(self.FG.feed_forward(input_vect), self.FH.feed_forward(h_prev)),
                                       self.output_activation, 'f'+str(self.timestamp))
        o = self.A.activation_function(np.add(self.OG.feed_forward(input_vect), self.OH.feed_forward(h_prev)),
                                       self.output_activation, 'o'+str(self.timestamp))
        s = np.add(np.multiply(g, i), np.multiply(s_prev, f))
        h = np.multiply(self.A.activation_function(s, self.hidden_activation, 'h' + str(self.timestamp)), o)
        self.Vectors['s'+str(self.timestamp)] = s
        self.Vectors['h'+str(self.timestamp)] = h
        self.Vectors['g'+str(self.timestamp)] = g
        self.Vectors['i'+str(self.timestamp)] = i
        self.Vectors['f'+str(self.timestamp)] = f
        self.Vectors['o'+str(self.timestamp)] = o
        self.Vectors['x'+str(self.timestamp)] = input_vect
        self.timestamp += 1
        return h

    # for I, G and F Weights
    def dhI(self, j, timestamp):
        return self.Vectors["o" + str(timestamp)][j] * self.A.errors["h" + str(timestamp)][j]

    def dcI(self, j, timestamp):
        return self.Vectors["g" + str(timestamp)][j] * self.A.errors["i" + str(timestamp)][j]

    def dI_w(self, dh_dc, dc_di, i, j, timestamp):
        dc = self.Vectors["f" + str(timestamp)][j] * self.dc_dWi[i][j] + dc_di
        dh = dh_dc * dc
        return dh, dc

    def dI_u(self, dh_dc, dc_di, i, j, timestamp):
        dc = self.Vectors["f" + str(timestamp)][j] * self.dc_dUi[i][j] + dc_di
        dh = dh_dc * dc
        return dh, dc

    def dI_b(self, dh_dc, dc_di, j, timestamp):
        dc = self.Vectors["f" + str(timestamp)][j] * self.dc_dBi[j] + dc_di
        dh = dh_dc * dc
        return dh, dc

    def dWi(self, error, dh, dc, i, j, timestamp):
        comp = (self.Vectors['x'+str(timestamp)][i] + self.IH.Weight[j][j]*self.dhp_dWi[i][j])
        dc *= comp
        dh *= comp
        self.dc_dWi[i][j] = dc
        self.dhp_dWi[i][j] = dh
        self.IG.Weight[i][j] += error*dh

    def dUi(self, error, dh, dc, j1, j2, timestamp):
        comp = (self.Vectors['h'+str(timestamp)][j1] + self.IH.Weight[j1][j2] * self.dhp_dUi[j1][j2])
        dc *= comp
        dh *= comp
        self.dc_dUi[j1][j2] = dc
        self.dhp_dUi[j1][j2] = dh
        self.IH.Weight[j1][j2] += error * dh

    def dBi(self, error, dh, dc, j):
        comp = (1 + self.IH.Weight[j][j]*self.dhp_dBi[j])
        dc *= comp
        dh *= comp
        self.dc_dBi[j] = dc
        self.dhp_dBi[j] = dh
        self.IG.Bias[j] += error * dh

    def dcF(self, j, timestamp):
        return self.Vectors["s" + str(timestamp - 1)][j] * self.A.errors["f" + str(timestamp)][j]

    def dF_w(self, dh_dc, dc_df, i, j, timestamp):
        dc = self.Vectors["f" + str(timestamp)][j]*self.dc_dWf[i][j] + dc_df
        dh = dh_dc * dc
        return dh, dc

    def dF_u(self, dh_dc, dc_df, i, j, timestamp):
        dc = self.Vectors["f" + str(timestamp)][j]*self.dc_dUf[i][j] + dc_df
        dh = dh_dc * dc
        return dh, dc

    def dF_b(self, dh_dc, dc_df, j, timestamp):
        dc = self.Vectors["f" + str(timestamp)][j]*self.dc_dBf[j] + dc_df
        dh = dh_dc * dc
        return dh, dc

    def dWf(self, error, dh, dc, i, j, timestamp):
        comp = self.Vectors['x'+str(timestamp)][i] + self.FH.Weight[j][j]*self.dhp_dWf[i][j]
        dc *= comp
        dh *= comp
        self.dc_dWf[i][j] = dc
        self.dhp_dWf[i][j] = dh
        self.FG.Weight[i][j] += error*dh

    def dUf(self, error, dh, dc, j1, j2, timestamp):
        comp = (self.Vectors['h' + str(timestamp)][j1] + self.FH.Weight[j1][j2] * self.dhp_dUf[j1][j2])
        dc *= comp
        dh *= comp
        self.dc_dUf[j1][j2] = dc
        self.dhp_dUf[j1][j2] = dh
        self.FH.Weight[j1][j2] += error * dh

    def dBf(self, error, dh, dc, j):
        comp = (1 + self.FH.Weight[j][j]*self.dhp_dBf[j])
        dc *= comp
        dh *= comp
        self.dc_dBf[j] = dc
        self.dhp_dBf[j] = dh
        self.FG.Bias[j] += error * dh

    def dcG(self, j, timestamp):
        return self.Vectors["i" + str(timestamp)][j] * self.A.errors["g" + str(timestamp)][j]

    def dG_w(self, dh_dc, dc_dg, i, j, timestamp):
        dc = self.Vectors["f" + str(timestamp)][j] * self.dc_dWg[i][j] + dc_dg
        dh = dh_dc * dc
        return dh, dc

    def dG_u(self, dh_dc, dc_dg, i, j, timestamp):
        dc = self.Vectors["f" + str(timestamp)][j] * self.dc_dUg[i][j] + dc_dg
        dh = dh_dc * dc
        return dh, dc

    def dG_b(self, dh_dc, dc_dg, j, timestamp):
        dc = self.Vectors["f" + str(timestamp)][j] * self.dc_dBg[j] + dc_dg
        dh = dh_dc * dc
        return dh, dc

    def dWg(self, error, dh, dc, i, j, timestamp):
        comp = (self.Vectors['x'+str(timestamp)][i] + self.GH.Weight[j][j] * self.dhp_dWg[i][j])
        dc *= comp
        dh *= comp
        self.dc_dWg[i][j] = dc
        self.dhp_dWg[i][j] = dh
        self.GG.Weight[i][j] += error*dh

    def dUg(self, error, dh, dc, j1, j2, timestamp):
        comp = (self.Vectors['h' + str(timestamp)][j1] + self.GH.Weight[j1][j2] * self.dhp_dUg[j1][j2])
        dc *= comp
        dh *= comp
        self.dc_dUg[j1][j2] = dc
        self.dhp_dUg[j1][j2] = dh
        self.GH.Weight[j1][j2] += error * dh

    def dBg(self, error, dh, dc, j):
        comp = (1 + self.GH.Weight[j][j] * self.dhp_dBg[j])
        dc *= comp
        dh *= comp
        self.dc_dBg[j] = dc
        self.dhp_dBg[j] = dh
        self.GG.Bias[j] += error * dh

    def dO(self, j, timestamp):
        dh = np.tanh(self.Vectors['s' + str(timestamp)])[j] * (self.A.errors["o" + str(timestamp)][j])
        return dh

    def dWo(self, error, dh, i, j, timestamp):
        dh *= (self.Vectors['x'+str(timestamp)][i] + self.OH.Weight[j][j] * self.dhp_dWo[i][j])
        self.dhp_dWo[i][j] = dh
        self.OG.Weight[i][j] += error * dh

    def dUo(self, error, dh, j1, j2, timestamp):
        dh *= (self.Vectors['h' + str(timestamp)][j1] + self.OH.Weight[j1][j2] * self.dhp_dUo[j1][j2])
        self.dhp_dUo[j1][j2] = dh
        self.OH.Weight[j1][j2] += error * dh

    def dBo(self, error, dh, j):
        dh *= (1 + self.OH.Weight[j][j] * self.dhp_dBo[j])
        self.dhp_dBo[j] = dh
        self.OG.Bias[j] += error * dh

    # def gradient_h(self, error, i, j, timestamp):
    #     for j in range(self.hidden_size):
    #         e = error_vect[j]
    #         dh, dhO = self.dhI(j, timestamp), self.dO(j, timestamp)
    #         dcI, dcG, dcF = self.dcI(j, timestamp), self.dcG(j, timestamp), self.dcF(j, timestamp)
    #
    #         if self.bias_bool:
    #             dih, dic = self.dI_b(dh, dcI, j, timestamp)
    #             dfh, dfc = self.dF_b(dh, dcF, j, timestamp)
    #             dgh, dgc = self.dI_b(dh, dcG, j, timestamp)
    #             self.dBi(e, dih, dic, j, training_rate)
    #             self.dBf(e, dfh, dfc, j, training_rate)
    #             self.dBg(e, dgh, dgc, j, training_rate)
    #             self.dBo(e, dhO, j, training_rate)
    #
    #         for i in range(self.input_size):
    #             dih, dic = self.dI_w(dh, dcI, i, j, timestamp)
    #             dfh, dfc = self.dF_w(dh, dcF, i, j, timestamp)
    #             dgh, dgc = self.dI_w(dh, dcG, i, j, timestamp)
    #             self.dWi(e, dih, dic, i, j, timestamp, training_rate)
    #             self.dWg(e, dgh, dgc, i, j, timestamp, training_rate)
    #             self.dWf(e, dfh, dfc, i, j, timestamp, training_rate)
    #             self.dWo(e, dhO, i, j, timestamp, training_rate)
    #
    #         for i in range(self.hidden_size):
    #             dih, dic = self.dI_u(dh, dcI, i, j, timestamp)
    #             dfh, dfc = self.dF_u(dh, dcF, i, j, timestamp)
    #             dgh, dgc = self.dI_u(dh, dcG, i, j, timestamp)
    #             self.dUi(e, dih, dic, i, j, timestamp, training_rate)
    #             self.dUg(e, dgh, dgc, i, j, timestamp, training_rate)
    #             self.dUf(e, dfh, dfc, i, j, timestamp, training_rate)
    #             self.dUo(e, dhO, i, j, timestamp, training_rate)
                
    def gradient_h(self, error_vect, timestamp):
        for j in range(self.hidden_size):
            e = error_vect[j]
            dh, dhO = self.dhI(j, timestamp), self.dO(j, timestamp)
            dcI, dcG, dcF = self.dcI(j, timestamp), self.dcG(j, timestamp), self.dcF(j, timestamp)

            if self.bias_bool:
                dih, dic = self.dI_b(dh, dcI, j, timestamp)
                dfh, dfc = self.dF_b(dh, dcF, j, timestamp)
                dgh, dgc = self.dI_b(dh, dcG, j, timestamp)
                self.dBi(e, dih, dic, j)
                self.dBf(e, dfh, dfc, j)
                self.dBg(e, dgh, dgc, j)
                self.dBo(e, dhO, j)

            for i in range(self.input_size):
                dih, dic = self.dI_w(dh, dcI, i, j, timestamp)
                dfh, dfc = self.dF_w(dh, dcF, i, j, timestamp)
                dgh, dgc = self.dI_w(dh, dcG, i, j, timestamp)
                self.dWi(e, dih, dic, i, j, timestamp)
                self.dWg(e, dgh, dgc, i, j, timestamp)
                self.dWf(e, dfh, dfc, i, j, timestamp)
                self.dWo(e, dhO, i, j, timestamp)

            for i in range(self.hidden_size):
                dih, dic = self.dI_u(dh, dcI, i, j, timestamp)
                dfh, dfc = self.dF_u(dh, dcF, i, j, timestamp)
                dgh, dgc = self.dI_u(dh, dcG, i, j, timestamp)
                self.dUi(e, dih, dic, i, j, timestamp)
                self.dUg(e, dgh, dgc, i, j, timestamp)
                self.dUf(e, dfh, dfc, i, j, timestamp)
                self.dUo(e, dhO, i, j, timestamp)

    # def gradient(self, error, i, j):
    #     timestamp = 1
    #     while timestamp < self.timestamp:
    #         max_timestamp = min(timestamp + self.iterations, self.timestamp)
    #         while timestamp < max_timestamp:
    #             self.gradient_h(error, i, j, timestamp)
    #             timestamp += 1
    #         self.reset()

    def train(self, error_vect):
        timestamp = 1
        while timestamp < self.timestamp:
            max_timestamp = min(timestamp + self.iterations, self.timestamp)
            while timestamp < max_timestamp:
                self.gradient_h(error_vect, timestamp)
                timestamp += 1
            print(timestamp, max_timestamp, self.timestamp)
            self.reset()
        print("---------")

    def reset(self):
        self.dhp_dWi = np.zeros(shape=(self.input_size, self.hidden_size))
        self.dhp_dWf = np.zeros(shape=(self.input_size, self.hidden_size))
        self.dhp_dWg = np.zeros(shape=(self.input_size, self.hidden_size))
        self.dhp_dWo = np.zeros(shape=(self.input_size, self.hidden_size))
        self.dhp_dUi = np.zeros(shape=(self.hidden_size, self.hidden_size))
        self.dhp_dUf = np.zeros(shape=(self.hidden_size, self.hidden_size))
        self.dhp_dUg = np.zeros(shape=(self.hidden_size, self.hidden_size))
        self.dhp_dUo = np.zeros(shape=(self.hidden_size, self.hidden_size))
        self.dhp_dBi = np.zeros(self.hidden_size)
        self.dhp_dBg = np.zeros(self.hidden_size)
        self.dhp_dBf = np.zeros(self.hidden_size)
        self.dhp_dBo = np.zeros(self.hidden_size)

        self.dc_dWi = np.zeros(shape=(self.input_size, self.hidden_size))
        self.dc_dUi = np.zeros(shape=(self.hidden_size, self.hidden_size))
        self.dc_dWf = np.zeros(shape=(self.input_size, self.hidden_size))
        self.dc_dUf = np.zeros(shape=(self.hidden_size, self.hidden_size))
        self.dc_dWg = np.zeros(shape=(self.input_size, self.hidden_size))
        self.dc_dUg = np.zeros(shape=(self.hidden_size, self.hidden_size))
        self.dc_dBi = np.zeros(self.hidden_size)
        self.dc_dBf = np.zeros(self.hidden_size)
        self.dc_dBg = np.zeros(self.hidden_size)

    def get_output(self):
        return super().get_output_abs('h')

    def transformLossTensor(self, loss_tensor):
        assert np.ndim(loss_tensor) == 1
        w = np.linalg.pinv(self.IG.Weight)
        return np.matmul(loss_tensor, w)

# 0.96 convergence to desired output with parameters: weight_param=(0,1), bias_params=(0,1), hidden_activation=softmax


def test():
    s, t = 0, 0
    for _ in range(1):
        i, o = 20, 10
        R = LSTMCell(o, i, weight_param=(0, 1), bias_params=(0, 1), hidden_activation='softmax')
        vects = [np.random.random(size=i) for j in range(i)]

        y = np.zeros(o)
        y[1] = 1

        R.feed_forward(vects)

        for i in range(30):
            error_vect = np.subtract(y, R.get_output())
            R.train(np.multiply(error_vect, 0.6))

            R.feed_forward(vects)

        t += 1
        if np.argmax(R.get_output()) == 1:
            s += 1

    return s / t


if __name__ == "__main__":
    # i, o = 15, 10
    # R = LSTMCell(o, i, weight_param=(0, 1), bias_params=(0, 1), hidden_activation='softmax')
    # vects = [np.random.random(size=i) for j in range(i)]
    #
    # y = np.zeros(o)
    # y[1] = 1
    # #
    # for vect in vects:
    #     R.feed_forward(vect)
    # #
    # print(R.get_output())
    # print(np.argmax(R.get_output()))
    #
    # for i in range(20):
    #     error_vect = np.subtract(y, R.get_output())
    #     R.gradient(np.multiply(error_vect, 0.8))
    #
    #     for vect in vects:
    #         R.feed_forward(vect)
    #
    # print(R.get_output())
    # print(np.argmax(R.get_output()))
    print(test())

