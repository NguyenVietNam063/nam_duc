import numpy as np
from collections import deque


class Line(object):
    """
    Đại diện cho một đường viền làn đường duy nhất.
    """

    def __init__(self, x, y, h, w):
        """
        Khởi tạo một đối tượng đường để cung cấp các điểm vạch.
        x : Mảng tọa độ x cho các pixel đại diện cho một đường kẻ.
        y: Mảng tọa độ y cho các pixel đại diện cho một đường.
        h : Chiều cao hình ảnh theo pixel.
        w : Chiều rộng hình ảnh tính bằng pixel.
        """
        # hệ số đa thức cho sự phù hợp gần đây nhất
        self.h = h
        self.w = w
        self.frame_impact = 0
        self.coefficients = deque(maxlen=5)
        self.process_points(x, y)

    def process_points(self, x, y):
        """
        Phù hợp với đa thức nếu có đủ điểm để thử và xấp xỉ một dòng và cập nhật một hàng đợi các hệ số.
        """
        enough_points = len(y) > 0 and np.max(y) - np.min(y) > self.h * .625
        if enough_points or len(self.coefficients) == 0:
            self.fit(x, y)

    def get_points(self):
        """
        Tạo điểm của dòng phù hợp nhất hiện tại.
        """
        y = np.linspace(0, self.h - 1, self.h)
        current_fit = self.averaged_fit()
        return np.stack((
            current_fit[0] * y ** 2 + current_fit[1] * y + current_fit[2],
            y
        )).astype(np.int).T

    def averaged_fit(self):
        """
        Trả về hệ số.
        """
        return np.array(self.coefficients).mean(axis=0)

    def fit(self, x, y):
        """
        Phù hợp với đa thức để cung cấp điểm và trả về hệ số của nó.
        """
        self.coefficients.append(np.polyfit(y, x, 2))

    def radius_of_curvature(self):
        """
        Tính bán kính độ cong của đường trong hệ tọa độ thế giới thực.
        """
        # Xác định chuyển đổi trong x và y từ không gian pixel sang mét
        ym_per_pix = 27 / 720  # met trên mỗi pixel theo kích thước y
        xm_per_pix = 3.7 / 700  # met trên mỗi pixel theo kích thước x
      
        points = self.get_points()
        y = points[:, 1]
        x = points[:, 0]
        fit_cr = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)
        return int(((1 + (2 * fit_cr[0] * 720 * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0]))

    def camera_distance(self):
        """
        Tính khoảng cách đến máy ảnh trong hệ tọa độ trong thế giới thực.
        """
        points = self.get_points()
        xm_per_pix = 3.7 / 700
        x = points[np.max(points[:, 1])][0]
        return np.absolute((self.w // 2 - x) * xm_per_pix)