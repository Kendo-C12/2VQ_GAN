import cv2
import numpy as np
import matplotlib.pyplot as plt

from makeNoise import simulate_lora_transmission

# Load image (grayscale for simplicity)
img = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)

noise_img = simulate_lora_transmission('test.jpg')

noise_img = np.array(noise_img.convert('L'))
# Apply mean filter (average blur)
mean_filtered = cv2.blur(noise_img, (3, 3))  # kernel size 3x3

# Apply median filter
median_filtered = cv2.medianBlur(noise_img, 3)  # kernel size 3

# Show results
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.title("Original")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title("Noise Filter")
plt.imshow(noise_img, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title("Mean Filter")
plt.imshow(mean_filtered, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title("Median Filter")
plt.imshow(median_filtered, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
