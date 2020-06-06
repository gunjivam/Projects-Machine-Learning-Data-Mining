from nn.cnn.Filters import Filters
from nn.cnn.Activation import ConvActivations
from nn.cnn.window import Window
import numpy as np
import time


class Convolution:
    def __init__(self, stride=2, window_dimensions=(3, 3, 1),
                 filter_params=(0, 2), bias_params=(0, 1), number_filters=3, new_filters=True, bias=True,
                 activation='relu', alpha=0.2, padding=0, training=True):
        self.win_height, self.win_width, self.win_depth = window_dimensions
        self.padding = padding
        self.stride = stride
        self.Filter = Filters(window_dimensions, filter_params, bias_params, number_filters, new_filters, bias)
        self.Window = Window(window_dimensions)
        self.Activation = ConvActivations(alpha=alpha, func=activation)
        self.conv_width, self.conv_height = (0, 0)
        self.gradients = [[] for _ in range(self.Filter.num)]
        self.train = training

    def convolute(self, image):
        image = np.asarray(image)
        assert np.ndim(image) >= 3
        # if image.ndim == 3:
        #     image = [i.flatten() for i in image]

        self.gradients = [[] for _ in range(self.Filter.num)]
        d, w, h = image.shape
        h += self.padding
        w += self.padding
        row = -self.padding
        layers = [[] for _ in range(self.Filter.num)]
        while row < h:
            rws = [[] for _ in range(self.Filter.num)]
            grws = [[] for _ in range(self.Filter.num)]
            col = -self.padding
            while col < w:
                win = self.Window.get_window(image, row, col)
                for f in range(self.Filter.num):
                    val = np.add(np.dot(win, self.Filter.Filters["f"+str(f)]), self.Filter.Filters["b"+str(f)])
                    val, dval = self.Activation.activation_function(val)
                    rws[f].append(val)
                    if self.train:
                        g = []
                        for wn in win:
                            g.append(wn*dval)
                        grws[f].append(g)

                col += self.stride
            for i in range(self.Filter.num):
                layers[i].append(rws[i])
                self.gradients[i].append(grws[i])
            row += self.stride
        self.conv_width, self.conv_height = np.shape(layers[0])
        return layers

    def gradient(self, error_mtx):
        rows, cols, fl = len(error_mtx[0]), len(error_mtx[0][0]), self.win_width*self.win_height*self.win_depth
        for i in range(rows):
            for j in range(cols):
                for n in range(self.Filter.num):
                    e = error_mtx[n][i][j]
                    self.Filter.Filters["b" + str(n)] += e
                    for f in range(fl):
                        # print("----------------------")
                        # print(self.Filter.Filters["f" + str(n)][f])
                        self.Filter.Filters["f"+str(n)][f] += e*self.gradients[n][i][j][f]
                        # print(e, self.gradients[n][i][j][f])
                        # print(self.Filter.Filters["f"+str(n)][f])

    def print(self, arr):
        for row in arr:
            print(row)
        print("<-------------->")


if __name__ == "__main__":
    C = Convolution(padding=0, window_dimensions=(3, 3, 3), stride=3, number_filters=4, activation='tanh')
    im = np.random.random((3, 15, 15))
    t = time.time()
    c1 = C.convolute(im)
    # print(len(C.Filter.Filters['f0']))
    print(np.asarray(c1).shape)
    # print(time.time() - t)
    print(np.asarray(C.gradients).shape)

    # print(c1)
    out = np.zeros((5, 5))
    out[0] = 2
    out[1] = 4
    out[2] = 0
    out[3] = 6
    out[4] = 8

    # for _ in range(1000):
    #     err = np.subtract(out, c1)*0.0005
    #     C.gradient(err)
    #     c1 = C.convolute(im)
    #     # print(np.asarray(c1).shape)
    #     # print(np.asarray(C.gradients).shape)
    #     # print("----------")
    #
    # for d in c1:
    #     for r in d:
    #         print(r)
    #     print("=====================================")
    #     break


