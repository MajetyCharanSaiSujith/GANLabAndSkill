import cv2
import numpy as np
import matplotlib.pyplot as plztelet
from skimage import img_as_float
from skimage.filters import sobel, prewitt

# Load Lena image (local) and convert to grayscale
lena_rgb = cv2.imread('img.png')  # Ensure lena.png is in the same folder
lena_gray = cv2.cvtColor(lena_rgb, cv2.COLOR_BGR2GRAY)

# ---------- Filtering ----------
# Gaussian Blur
gaussian_blur = cv2.GaussianBlur(lena_gray, (5, 5), 0)

# Median Filter
median_blur = cv2.medianBlur(lena_gray, 5)

# ---------- Edge Detection ----------
# Sobel
sobel_edges = sobel(img_as_float(lena_gray))

# Prewitt
prewitt_edges = prewitt(img_as_float(lena_gray))

# Canny
canny_edges = cv2.Canny(lena_gray, 100, 200)

# ---------- Display ----------
titles = ['Original (Gray)', 'Gaussian Blur', 'Median Blur',
          'Sobel', 'Prewitt', 'Canny']
images = [lena_gray, gaussian_blur, median_blur,
          sobel_edges, prewitt_edges, canny_edges]

plt.figure(figsize=(12, 8))
for i in range(len(images)):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()