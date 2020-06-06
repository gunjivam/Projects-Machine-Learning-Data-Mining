import numpy as np
import math


class Flatten:
    def __init__(self, layers):
        self.vector = self.flatten(layers)

    @staticmethod
    def flatten(layers):
        vect = []
        for layer in layers:
            for row in layer:
                vect.extend(row)
        return np.asarray(vect)

    def convert_index(self, index, rows, cols):
        d = math.floor(index/(rows*cols))
        r = math.floor((index - d*rows*cols)/ cols)
        c = d - r*cols - d*rows*cols
        return d, r, c
