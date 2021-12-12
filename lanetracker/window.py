import numpy as np


class Window(object):
    """
    Đại diện cho một cửa sổ quét được sử dụng để phát hiện các điểm có khả năng đại diện cho các đường viền làn đường.
    """

    def __init__(self, y1, y2, x, m=100, tolerance=50):
        """
        Khởi tạo một đối tượng cửa sổ.
        tolerance: Số điểm ảnh tối thiểu chúng ta cần phát hiện trong cửa sổ để điều chỉnh tọa độ x của nó.
        """
        self.x = x
        self.mean_x = x
        self.y1 = y1
        self.y2 = y2
        self.m = m
        self.tolerance = tolerance

    def pixels_in(self, nonzero, x=None):
        """
        Trả về chỉ số của các pixel trong 'nonzero' nằm trong cửa sổ này.
        nonzero : Tọa độ của các pixel không bằng không trong hình ảnh.
        """
        if x is not None:
            self.x = x
        win_indices = (
            (nonzero[0] >= self.y1) & (nonzero[0] < self.y2) &
            (nonzero[1] >= self.x - self.m) & (nonzero[1] < self.x + self.m)
        ).nonzero()[0]
        if len(win_indices) > self.tolerance:
            self.mean_x = np.int(np.mean(nonzero[1][win_indices]))
        else:
            self.mean_x = self.x

        return win_indices

    def coordinates(self):
        """
        Trả về tọa độ.
        """
        return ((self.x - self.m, self.y1), (self.x + self.m, self.y2))