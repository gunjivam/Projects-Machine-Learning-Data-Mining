from nn.cnn.window import Window
import numpy as np


class MaxPooling:
    gradients = []

    def __init__(self, width, height, pool_size=4):
        self.pool_size = pool_size
        self.W = Window((pool_size, pool_size))
        self.inp_width, self.inp_height = width, height
        self.__set_ouput_dimensions__(width, height)

    def __set_ouput_dimensions__(self, width, height):
        self.pool_width, self.pool_height = int(width / self.pool_size), int(height / self.pool_size)

    def pool(self, layers):
        new_layers = []
        for layer in layers:
            gradient = np.zeros((self.inp_height, self.inp_width))
            new_layer = np.zeros((self.pool_height, self.pool_width))
            row = 0
            for _row in range(self.pool_height):
                col = 0
                for _col in range(self.pool_width):
                    num, pos = self.W.get_max_2Dwindow(layer, row, col)
                    new_layer[_row, _col] = num
                    gradient[row + int((pos / self.pool_size)), col + (pos % self.pool_size)] = 1
                    col += self.pool_size
                row += self.pool_size
            self.gradients.append(gradient)
            new_layers.append(new_layer)
        return new_layers


if __name__ == "__main__":
    sz = 6
    I = [np.random.random((sz, sz)), np.random.random((sz, sz))]
    M = MaxPooling(sz, sz, 3)
    M.pool(I)
