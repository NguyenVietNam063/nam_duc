import numpy as np
import cv2
from lanetracker.window import Window
from lanetracker.line import Line
from lanetracker.gradients import get_edges
from lanetracker.perspective import flatten_perspective


class LaneTracker(object):
    """
    Theo dõi làn đường trong một loạt các khung liên tiếp.
    """

    def __init__(self, first_frame, n_windows=9):
    
        """
        Khởi tạo một đối tượng theo dõi.
        first_frame: Khung đầu tiên của loạt khung. Chúng tôi sử dụng nó để lấy kích thước và khởi tạo giá trị.
        n_windows: Số lượng cửa sổ sử dụng để theo dõi mỗi cạnh làn đường.
        """
        (self.h, self.w, _) = first_frame.shape
        self.win_n = n_windows
        self.left = None
        self.right = None
        self.l_windows = []
        self.r_windows = []
        self.initialize_lines(first_frame)

    def initialize_lines(self, frame):
    
        """
        Tìm điểm bắt đầu cho các đường bên trái và bên phải (ví dụ: cạnh làn đường) và khởi chạy các đối tượng Cửa sổ và Đường.
        frame: Khung để quét các mép làn.
        """
        # Chụp biểu đồ của nửa dưới của hình ảnh
        edges = get_edges(frame)
        (flat_edges, _) = flatten_perspective(edges)
        histogram = np.sum(flat_edges[int(self.h / 2):, :], axis=0)

        nonzero = flat_edges.nonzero()
        # Tạo danh sách trống để nhận chỉ số pixel làn bên trái và bên phải
        l_indices = np.empty([0], dtype=np.int)
        r_indices = np.empty([0], dtype=np.int)
        window_height = int(self.h / self.win_n)

        for i in range(self.win_n):
            l_window = Window(
                y1=self.h - (i + 1) * window_height,
                y2=self.h - i * window_height,
                x=self.l_windows[-1].x if len(self.l_windows) > 0 else np.argmax(histogram[:self.w // 2])
            )
            r_window = Window(
                y1=self.h - (i + 1) * window_height,
                y2=self.h - i * window_height,
                x=self.r_windows[-1].x if len(self.r_windows) > 0 else np.argmax(histogram[self.w // 2:]) + self.w // 2
            )
            # Nối các chỉ số khác không trong ranh giới cửa sổ vào danh sách
            l_indices = np.append(l_indices, l_window.pixels_in(nonzero), axis=0)
            r_indices = np.append(r_indices, r_window.pixels_in(nonzero), axis=0)
            self.l_windows.append(l_window)
            self.r_windows.append(r_window)
        self.left = Line(x=nonzero[1][l_indices], y=nonzero[0][l_indices], h=self.h, w = self.w)
        self.right = Line(x=nonzero[1][r_indices], y=nonzero[0][r_indices], h=self.h, w = self.w)

    def scan_frame_with_windows(self, frame, windows):
    
        """
        Quét khung bằng cách sử dụng các cửa sổ đã khởi tạo để cố gắng theo dõi các mép làn đường.
        """
        indices = np.empty([0], dtype=np.int)
        nonzero = frame.nonzero()
        window_x = None
        for window in windows:
            indices = np.append(indices, window.pixels_in(nonzero, window_x), axis=0)
            window_x = window.mean_x
        return (nonzero[1][indices], nonzero[0][indices])

    def process(self, frame, draw_lane=True, draw_statistics=True):
    
        """
        Theo dõi làn đường đầy đủ trên một khung.
            frame: Khung mới để xử lý.
            draw_lane: Cho biết cần vẽ làn đường trên đầu khung.
            draw_statistics: Cho biết có cần hiển thị thông tin gỡ lỗi trên đầu khung hay không.
        """
        edges = get_edges(frame)
        (flat_edges, unwarp_matrix) = flatten_perspective(edges)
        (l_x, l_y) = self.scan_frame_with_windows(flat_edges, self.l_windows)
        self.left.process_points(l_x, l_y)
        (r_x, r_y) = self.scan_frame_with_windows(flat_edges, self.r_windows)
        self.right.process_points(r_x, r_y)

        if draw_statistics:
            edges = get_edges(frame, separate_channels=True)
            debug_overlay = self.draw_debug_overlay(flatten_perspective(edges)[0])
            top_overlay = self.draw_lane_overlay(flatten_perspective(frame)[0])
            debug_overlay = cv2.resize(debug_overlay, (0, 0), fx=0.3, fy=0.3)
            top_overlay = cv2.resize(top_overlay, (0, 0), fx=0.3, fy=0.3)
            frame[:250, :, :] = frame[:250, :, :] * .4
            (h, w, _) = debug_overlay.shape
            frame[20:20 + h, 20:20 + w, :] = debug_overlay
            frame[20:20 + h, 20 + 20 + w:20 + 20 + w + w, :] = top_overlay
            text_x = 20 + 20 + w + w + 20
            #self.draw_text(frame, 'Radius of curvature:  {} m'.format(self.radius_of_curvature()), text_x, 80)
            #self.draw_text(frame, 'Distance (left):       {:.1f} m'.format(self.left.camera_distance()), text_x, 140)
            #self.draw_text(frame, 'Distance (right):      {:.1f} m'.format(self.right.camera_distance()), text_x, 200)

        if draw_lane:
            frame = self.draw_lane_overlay(frame, unwarp_matrix)

        return frame

    def draw_text(self, frame, text, x, y):
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)

    def draw_debug_overlay(self, binary, lines=True, windows=True):
    
        """
        Vẽ một lớp phủ với thông tin trên chế độ bird's-eye view toàn cảnh của con đường 
            binary: Khung thành lớp phủ.
            lines: Chỉ ra nếu cần vẽ đường.
            windows: Cho biết nếu cần vẽ các cửa sổ.
        """
        if len(binary.shape) == 2:
            image = np.dstack((binary, binary, binary))
        else:
            image = binary
        if windows:
            for window in self.l_windows:
                coordinates = window.coordinates()
                cv2.rectangle(image, coordinates[0], coordinates[1], (1., 1., 0), 2)
            for window in self.r_windows:
                coordinates = window.coordinates()
                cv2.rectangle(image, coordinates[0], coordinates[1], (1., 1., 0), 2)
        if lines:
            cv2.polylines(image, [self.left.get_points()], False, (1., 0, 0), 2)
            cv2.polylines(image, [self.right.get_points()], False, (1., 0, 0), 2)
        return image * 255

    def draw_lane_overlay(self, image, unwarp_matrix=None):
    
        """
        Vẽ một lớp phủ với làn đường được theo dõi áp dụng phối cảnh unwarp để chiếu nó lên khung ban đầu.
        image: Khung nguyên bản.
        unwarp_matrix: Chuyển đổi ma trận để hủy chế độ bird's eye view về khung hình ban đầu. 
        """
        # Tạo một hình ảnh để vẽ các đường trên đó
        overlay = np.zeros_like(image).astype(np.uint8)
        points = np.vstack((self.left.get_points(), np.flipud(self.right.get_points())))
        # Vẽ làn đường vào hình trống bị cong
        cv2.fillPoly(overlay, [points], (0, 255, 0))
        if unwarp_matrix is not None:
            # Trở lại không gian hình ảnh ban đầu
            overlay = cv2.warpPerspective(overlay, unwarp_matrix, (image.shape[1], image.shape[0]))
        # Kết hợp kết quả với hình ảnh gốc
        return cv2.addWeighted(image, 1, overlay, 0.3, 0)

    def radius_of_curvature(self):
    
        """
        Tính bán kính cong của làn đường bằng cách lấy trung bình độ cong của các đường biên.
        """
        return int(np.average([self.left.radius_of_curvature(), self.right.radius_of_curvature()]))