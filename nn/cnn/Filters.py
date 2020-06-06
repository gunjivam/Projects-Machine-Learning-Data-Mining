import numpy as np
from random import uniform


class Filters:
    def __init__(self, filter_dimensions, filter_params=(0, 2), bias_params=(0, 1),
                 number_filters=3, new_filters=True, bias=True):
        self.f_height, self.f_width, self.f_depth = filter_dimensions
        self.f_min, self.f_max = filter_params
        self.num = number_filters
        self.b_min, self.b_max = bias_params
        self.Filters = {}
        if not bias:
            self.b_min, self.b_max = 0, 0
        if new_filters:
            self.__create_filters__()

    def __create_filters__(self):
        for i in range(self.num):
            self.Filters['f'+str(i)] = np.random.uniform(self.f_min, self.f_max, (self.f_height*self.f_width*self.f_depth))
            self.Filters['b'+str(i)] = uniform(self.b_min, self.b_max)


if __name__ == '__main__':
    f = Filters((2,2))
    print(f.Filters)
