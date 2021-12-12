import numpy as np
import cv2


def gradient_abs_value_mask(image, sobel_kernel=3, axis='x', threshold=(0, 255)):

    """
    Tạo mặt nạ cho hình ảnh dựa trên giá trị tuyệt đối của gradient.
        image: Hình ảnh để che.
        sobel_kernel: Kernel of the Sobel gradient operation.
        axis: Trục của gradient, 'x' hoặc 'y'.
        threshold: Ngưỡng giá trị để nó xuất hiện trong mặt nạ.
    """
    # Lấy giá trị tuyệt đối của đạo hàm theo x hoặc y cho trước
    if axis == 'x':
        sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if axis == 'y':
        sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Quy mô thành 8-bit (0 - 255) sau đó chuyển đổi thành:
    sobel = np.uint8(255 * sobel / np.max(sobel))
    # Tạo mặt nạ
    mask = np.zeros_like(sobel)
    mask[(sobel >= threshold[0]) & (sobel <= threshold[1])] = 1
    return mask

def gradient_magnitude_mask(image, sobel_kernel=3, threshold=(0, 255)):

    """
    Tạo mặt nạ cho hình ảnh dựa trên độ lớn của gradient.
    """
    # Lấy gradient theo x và y riêng biệt
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Tính độ lớn
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    # Quy mô thành 8-bit (0 - 255) sau đó chuyển đổi thành:
    magnitude = (magnitude * 255 / np.max(magnitude)).astype(np.uint8)
    # Tạo mặt nạ nhị phân trong đó các ngưỡng hướng được đáp ứng
    mask = np.zeros_like(magnitude)
    mask[(magnitude >= threshold[0]) & (magnitude <= threshold[1])] = 1
    return mask

def gradient_direction_mask(image, sobel_kernel=3, threshold=(0, np.pi / 2)):

    """
    Tạo mặt nạ cho hình ảnh dựa trên hướng gradient.
    """
    # Lấy gradient theo x và y riêng biệt    
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Lấy giá trị tuyệt đối của gradient x và y và tính hướng của gradient
    direction = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))
    # Tạo mặt nạ nhị phân trong đó các ngưỡng hướng được đáp ứng
    mask = np.zeros_like(direction)
    mask[(direction >= threshold[0]) & (direction <= threshold[1])] = 1
    return mask

def color_threshold_mask(image, threshold=(0, 255)):

    """
    Tạo mặt nạ cho hình ảnh dựa trên cường độ màu.
    """
    mask = np.zeros_like(image)
    mask[(image > threshold[0]) & (image <= threshold[1])] = 1
    return mask

def get_edges(image, separate_channels=False):

    """
    Tạo mặt nạ cho hình ảnh dựa trên thành phần của bộ dò cạnh: giá trị gradient,
    độ lớn của gradient, hướng và màu sắc của gradient.
    """
    # Chuyển đổi sang không gian màu HLS
    hls = cv2.cvtColor(np.copy(image), cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hls[:, :, 2]
    # Nhận kết hợp tất cả các mặt nạ ngưỡng gradient
    gradient_x = gradient_abs_value_mask(s_channel, axis='x', sobel_kernel=3, threshold=(20, 100))
    gradient_y = gradient_abs_value_mask(s_channel, axis='y', sobel_kernel=3, threshold=(20, 100))
    magnitude = gradient_magnitude_mask(s_channel, sobel_kernel=3, threshold=(20, 100))
    direction = gradient_direction_mask(s_channel, sobel_kernel=3, threshold=(0.7, 1.3))
    gradient_mask = np.zeros_like(s_channel)
    gradient_mask[((gradient_x == 1) & (gradient_y == 1)) | ((magnitude == 1) & (direction == 1))] = 1
    # Nhận mặt nạ ngưỡng màu
    color_mask = color_threshold_mask(s_channel, threshold=(170, 255))

    if separate_channels:
        return np.dstack((np.zeros_like(s_channel), gradient_mask, color_mask))
    else:
        mask = np.zeros_like(gradient_mask)
        mask[(gradient_mask == 1) | (color_mask == 1)] = 1
        return mask