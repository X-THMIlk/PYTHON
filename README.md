chỉnh sáng 
import cv2
import numpy as np
import matplotlib.pyplot as plt


def min_max_normalization(image):
    I_min, I_max = np.min(image), np.max(image)
    normalized = ((image - I_min) / (I_max - I_min) * 255).astype(np.uint8)
    return normalized


def lmin_lmax_normalization(image, percentile=1):
    L_min = np.percentile(image, percentile)  # L_min loại bỏ nhiễu thấp
    L_max = np.percentile(image, 100 - percentile)  # L_max loại bỏ nhiễu cao

    image_clipped = np.clip(image, L_min, L_max)  # Cắt giá trị ngoài khoảng L_min - L_max
    normalized = ((image_clipped - L_min) / (L_max - L_min) * 255).astype(np.uint8)
    return normalized


def adjust_brightness(image, value):
    """Tăng/Giảm độ sáng ảnh bằng cách cộng/trừ giá trị."""
    adjusted = np.clip(image.astype(np.int16) + value, 0, 255).astype(np.uint8)
    return adjusted


# Đọc ảnh màu
image = cv2.imread('E:\\ThucHanh23032025\\a3.jpg')

# Áp dụng hai phương pháp
min_max_img = min_max_normalization(image)
lmin_lmax_img = lmin_lmax_normalization(image)

# Tăng/Giảm độ sáng
brighter_img = adjust_brightness(image, 50)  # Tăng sáng

darker_img = adjust_brightness(image, -50)  # Giảm sáng

# Hiển thị kết quả
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original')
axes[1].imshow(cv2.cvtColor(min_max_img, cv2.COLOR_BGR2RGB))
axes[1].set_title('Min-Max Normalization')
axes[2].imshow(cv2.cvtColor(lmin_lmax_img, cv2.COLOR_BGR2RGB))
axes[2].set_title('Lmin-Lmax Normalization')
axes[3].imshow(cv2.cvtColor(brighter_img, cv2.COLOR_BGR2RGB))
axes[3].set_title('Brighter Image')
axes[4].imshow(cv2.cvtColor(darker_img, cv2.COLOR_BGR2RGB))
axes[4].set_title('Darker Image')

for ax in axes:
    ax.axis('off')
plt.show()
histogram
import cv2
import matplotlib.pyplot as plt

# Đọc ảnh
image = cv2.imread('E:\\ThucHanh23032025\\a3.jpg')

# Chuyển ảnh sang ảnh xám (grayscale)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Tính toán Histogram
histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

# Hiển thị Histogram
plt.plot(histogram)
plt.title('Histogram of Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()
tuong phan
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh
image = cv2.imread('E:\\ThucHanh23032025\\a3.jpg')

# Chuyển ảnh sang ảnh xám (grayscale) để dễ dàng quan sát sự thay đổi
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Tăng/giảm độ tương phản: Nhân giá trị pixel với alpha
# alpha > 1: Tăng độ tương phản, alpha < 1: Giảm độ tương phản
contrast_value = 2.0  # Thay đổi giá trị alpha để điều chỉnh độ tương phản
contrasted_image = cv2.convertScaleAbs(gray_image, alpha=contrast_value, beta=0)

# Hiển thị ảnh gốc và ảnh đã thay đổi độ tương phản
plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(contrasted_image, cmap='gray')
plt.title('Contrasted Image')
plt.show()

# Lưu ảnh đã thay đổi độ tương phản nếu cần
cv2.imwrite('contrasted_image.jpg', contrasted_image)
