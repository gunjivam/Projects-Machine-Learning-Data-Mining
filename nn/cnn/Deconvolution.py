import numpy as np
from nn.cnn.Filters import Filters
from nn.cnn.Activation import ConvActivations
import math


class Deconvolute:
    def __init__(self, filter_dimensions=(3, 3, 3), filter_params=(0, 2), bias_params=(0, 1), stride = 2,
                 number_filters=3, new_filters=True, bias=True, activation='relu', alpha=0.2, training=True):
        self.win_height, self.win_width, self.win_depth = filter_dimensions
        self.Filter = Filters(filter_dimensions, filter_params, bias_params, number_filters, new_filters, bias)
        self.Activation = ConvActivations(alpha=alpha, func=activation)
        self.conv_width, self.conv_height = (0, 0)
        self.train = training
        self.stride = stride
        self.gradients = [[] for _ in range(self.Filter.num)]

    def f(self, img, flt, stride, fwidth, fheight, fi=0):
        ix, iy, fl = len(img[0]), len(img), len(flt)
        if fwidth == ix:
            sx = ix*2-1 + (ix-1)*(stride-1)
            sy = iy*2-1 + (iy-1)*(stride-1)

        else:
            sx = (fwidth + ix - 1) + (stride-1)*(ix-1)
            sy = (fheight + iy - 1) + (stride-1)*(iy-1)

        res = np.zeros((sy, sx))
        g = [[[0 for _ in range(fl)] for _ in range(sx)] for _ in range(sy)]
        r, c = 0, 0
        for ii in range(iy):
            for ij in range(ix):
                e = img[ii][ij]
                tr, tc = r, c
                for f in range(fl):
                    val, dval = self.Activation.activation_function(e*flt[f])
                    res[tr][tc] += val
                    g[tr][tc][f] = e
                    tc += 1
                    if tc >= c + fwidth:
                        tc = c
                        tr += 1
                    if tr >= r + fheight:
                        tr = r
                c += stride

            c = 0
            r += stride
        self.gradients[fi].append(g)
        return res

    def deconvolute(self, image):
        image = np.asarray(image)
        assert np.ndim(image) == 3 and self.win_depth == len(image)
        layers = []
        for f in range(self.Filter.num):
            fltr = np.asarray(self.Filter.Filters["f"+str(f)]).reshape((self.win_depth, self.win_height*self.win_width))
            ls = self.f(image[0], fltr[0], self.stride, self.win_width, self.win_height, f)

            for ly in range(1, self.win_depth):
                ls = np.add(self.f(image[ly], fltr[ly], self.stride, self.win_width, self.win_height, f), ls)

            layers.append(np.add(ls, self.Filter.Filters["b"+str(f)]))
        return layers

    def gradient(self, error_mtx):
        rows, cols, fl = len(error_mtx[0]), len(error_mtx[0][0]), self.win_width * self.win_height
        for i in range(rows):
            for j in range(cols):
                for n in range(self.Filter.num):
                    e = error_mtx[n][i][j]
                    self.Filter.Filters["b" + str(n)] += e
                    for d in range(self.win_depth):
                        for f in range(fl):
                            self.Filter.Filters["f" + str(n)][fl*d + f] += e * self.gradients[n][d][i][j][f]


if __name__ == "__main__":
    D = Deconvolute(stride=2, filter_dimensions=(2, 2, 3), bias_params=(0, 0), filter_params=(1, 1), number_filters=2)
    n, m = 9, 9
    f = [1 for _ in range(n)]
    # f = [1, 2, 3, 4]
    i = np.reshape([j for j in range(1, m+1)], (int(math.sqrt(m)), int(math.sqrt(m))))
    # i = [[1, 2, 3], [4, 5, 6]]
    img = [i for _ in range(3)]

    # assert len(f) == n
    # # print(np.asarray(D.f(i, f, 1, int(math.sqrt(n)), int(math.sqrt(n)))).shape)
    lyrs = D.deconvolute(img)
    # print(np.asarray(lyrs).shape)
    # for lyr in lyrs:
    #     print(lyr)
    print(np.asarray(D.gradients).shape)
    #
    # out = np.zeros((6, 6))
    # out[0] = 20
    # out[1] = 40
    # out[2] = 80
    # out[3] = 60
    # out[4] = 80
    # out[5] = 100
    # # out[6] = 12
    #
    # # print(D.gradients)
    # for _ in range(200):
    #     err = np.subtract(out, lyrs)*0.0005
    #     # print(err[0])
    #     # print(lyrs[0])
    #     # print(out)
    #     # print("---------")
    #     D.gradient(err)
    #     lyrs = D.deconvolute(img)
    #     # print(np.asarray(c1).shape)
    #     # print(np.asarray(C.gradients).shape)
    #     # print("----------")
    #
    # print(out)
    # print(lyrs[0])
    # print(D.Filter.Filters)
    # # for d in c1:
    # #     for r in d:
    # #         print(r)
    # #     print("=====================================")
    # #     break
