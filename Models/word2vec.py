from nn.Dense import Dense
from Optimizer.Losses import Losses
import numpy as np


class Word2Vec:
    def __init__(self, i, activation="softmax", weight_params=(-1, 1), bias_params=(0, 1),
                 loss_function="cross_entropy"):
        self.i = i
        self.activation = activation
        self.Vectors = {}
        self.D = Dense(i, i, activation, weight_params, bias_params)
        self.L = Losses()
        self.loss_function = loss_function

    def f(self, vectors, target):
        for vector in vectors:
            z = np.asarray(self.D.feed_forward(vector))
            l = self.L.loss_function(self.loss_function, target, z)
            self.D.train(l[0])

    def run(self, vectors, context_size=2):
        for i in range(len(vectors)):
            left = i - context_size if i - context_size >= 0 else 0 if i != 0 else -1
            right = i + context_size if i + context_size < len(vectors) else len(vectors)-1 if i != len(vectors)-1 else \
                -1
            vects = []
            if left != -1:
                vects.extend(vectors[left:i])
            if right != -1:
                vects.extend(vectors[i+1:right+1])
            self.f(vects, np.asarray(vectors[i]))
            print(vectors[i], vects)

    def feed_forward(self, vector):
        return self.D.feed_forward(vector)


if __name__ == "__main__":
    i, n = 5, 5
    w = Word2Vec(i)
    inp = [[0 if k != j else 1 for k in range(i)] for j in range(n)]
    w.run(inp, context_size=2)
    print(w.feed_forward(inp[0]))

