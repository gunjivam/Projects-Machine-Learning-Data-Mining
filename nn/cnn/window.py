import numpy as np


class Window:
    def __init__(self, window_dimensions=(3, 3, 1)):
        if len(window_dimensions) == 3:
            self.win_cols, self.win_rows, self.win_depth = window_dimensions
        else:
            self.win_cols, self.win_rows = window_dimensions

    def get_window(self, array3d, start_row, start_col):
        assert array3d.ndim == 3
        window = np.zeros((self.win_rows * self.win_cols*self.win_depth))
        for d in range(self.win_depth):
            for row in range(self.win_rows):
                r = row + start_row
                if r >= 0:
                    for col in range(self.win_cols):
                        if col+start_col >= 0:
                            try:
                                window[(row*self.win_cols)+col + (d*self.win_cols*self.win_rows)] = array3d[d, r, col+start_col]
                            except IndexError:
                                break
        return window

    def get_max_2Dwindow(self, array2d, start_row, start_col):
        assert array2d.ndim == 2
        mx, pos = 0, 0
        for row in range(self.win_rows):
            r = row + start_row
            if r >= 0:
                for col in range(self.win_cols):
                    if col + start_col >= 0:
                        try:
                            elem = array2d[r, col + start_col]
                            if elem > mx:
                                mx = elem
                                pos = (row*self.win_cols)+col
                        except IndexError:
                            break
        return mx, pos

    @staticmethod
    def get_window_axis3(array3d, row, col):
        assert array3d.ndim == 3
        win = []
        for i in range(len(array3d)):
            win.append(array3d[i][row][col])
        return win


if __name__ == '__main__':
    z = [i for i in range(32)]
    z = np.reshape(z, (2, 4, 4))
    c = Window((3, 3, 2))
    w = c.get_window(z, 1, 2)
    print(z)
    print(w)
    pass
