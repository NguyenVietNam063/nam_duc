import numpy as np
import cv2


def flatten_perspective(image):
    """
    Làm cong hình ảnh từ camera phía trước của xe ánh xạ đường hte sang phối cảnh bird view.
    """
    # Nhận kích thước hình ảnh
    (h, w) = (image.shape[0], image.shape[1])
    # Xác định điểm nguồn
    source = np.float32([[w // 2 - 76, h * .625], [w // 2 + 76, h * .625], [-100, h], [w + 100, h]])
    # Xác định điểm đích tương ứng
    destination = np.float32([[100, 0], [w - 100, 0], [100, h], [w - 100, h]])
    transform_matrix = cv2.getPerspectiveTransform(source, destination)
    unwarp_matrix = cv2.getPerspectiveTransform(destination, source)
    return (cv2.warpPerspective(image, transform_matrix, (w, h)), unwarp_matrix)