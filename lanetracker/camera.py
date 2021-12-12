import numpy as np
import cv2
import matplotlib.image as mpimg


class CameraCalibration(object):
    """Chuẩn bị đường ống hiệu chuẩn máy ảnh dựa trên một tập hợp các hình ảnh hiệu chuẩn."""
    def __init__(self, calibration_images, pattern_size=(9, 6), retain_calibration_images=False):
        """
        Khởi tạo hiệu chuẩn máy ảnh dựa trên một tập hợp các hình ảnh hiệu chuẩn.
            calibration_images: Hình ảnh hiệu chuẩn.
            pattern_size: Hình dạng của mẫu hiệu chuẩn. 
            retain_calibration_images: Cho biết liệu chúng ta có cần giữ lại các hình ảnh hiệu chuẩn hay không.
        """
        self.camera_matrix = None
        self.dist_coefficients = None
        self.calibration_images_success = []
        self.calibration_images_error = []
        self.calculate_calibration(calibration_images, pattern_size, retain_calibration_images)

    def __call__(self, image):
    
        """Hiệu chỉnh hình ảnh dựa trên cài đặt đã lưu."""
        if self.camera_matrix is not None and self.dist_coefficients is not None:
            return cv2.undistort(image, self.camera_matrix, self.dist_coefficients, None, self.camera_matrix)
        else:
            return image

    def calculate_calibration(self, images, pattern_size, retain_calibration_images):
        """Chuẩn bị các cài đặt hiệu chuẩn."""
        # Chuẩn bị các điểm đối tượng
        pattern = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
        pattern[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        pattern_points = []  # Điểm 3d trong không gian thế giới thực
        image_points = []  # Điểm 2d trong mặt phẳng hình ảnh
        image_size = None

        # Tìm kiếm các góc
        for i, path in enumerate(images):
            image = mpimg.imread(path)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # Tìm các góc
            found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
            # Nếu tìm thấy, thêm điểm đối tượng và điểm hình ảnh
            if found:
                pattern_points.append(pattern)
                image_points.append(corners)
                image_size = (image.shape[1], image.shape[0])
                if retain_calibration_images:
                    cv2.drawChessboardCorners(image, pattern_size, corners, True)
                    self.calibration_images_success.append(image)
            else:
                if retain_calibration_images:
                    self.calibration_images_error.append(image)

        if pattern_points and image_points:
            _, self.camera_matrix, self.dist_coefficients, _, _ = cv2.calibrateCamera(
                pattern_points, image_points, image_size, None, None)