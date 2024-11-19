# Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the color image in OpenCV format (already uploaded as 'car.jpg')
image_path = 'Image_for_TeamSV/car.jpg'
color_image = cv2.imread(image_path)  # OpenCV reads in BGR format

# Convert to grayscale for grayscale testing
gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

# Define filters
# Mean (Averaging) filter
mean_filter_3x3 = cv2.blur(color_image, (3, 3))
mean_filter_9x9 = cv2.blur(color_image, (9, 9))

# Gaussian filter
gaussian_filter_3x3 = cv2.GaussianBlur(color_image, (3, 3), 0)
gaussian_filter_9x9 = cv2.GaussianBlur(color_image, (9, 9), 0)

# Sharpening filter - create kernel and apply
sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpened_image_3x3 = cv2.filter2D(color_image, -1, sharpening_kernel)
# For sharpening, 9x9 is not typical, so we only use 3x3

# Sobel edge detection
sobel_x_3x3 = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y_3x3 = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
sobel_x_9x9 = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=9)
sobel_y_9x9 = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=9)

median_3x3 = cv2.medianBlur(color_image, 3)
median_9x9 = cv2.medianBlur(color_image, 9)

# Combine Sobel x and y results for visualization
sobel_combined_3x3 = cv2.magnitude(sobel_x_3x3, sobel_y_3x3)
sobel_combined_9x9 = cv2.magnitude(sobel_x_9x9, sobel_y_9x9)

# Display the results
plt.figure(figsize=(12, 12))

# Original Color Image
plt.subplot(4, 3, 1)
plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
plt.title("Original Color Image")
plt.axis("off")

# Mean Filter
plt.subplot(4, 3, 2)
plt.imshow(cv2.cvtColor(mean_filter_3x3, cv2.COLOR_BGR2RGB))
plt.title("Mean Filter 3x3")
plt.axis("off")

plt.subplot(4, 3, 3)
plt.imshow(cv2.cvtColor(mean_filter_9x9, cv2.COLOR_BGR2RGB))
plt.title("Mean Filter 9x9")
plt.axis("off")

# Gaussian Filter
plt.subplot(4, 3, 4)
plt.imshow(cv2.cvtColor(gaussian_filter_3x3, cv2.COLOR_BGR2RGB))
plt.title("Gaussian Filter 3x3")
plt.axis("off")

plt.subplot(4, 3, 5)
plt.imshow(cv2.cvtColor(gaussian_filter_9x9, cv2.COLOR_BGR2RGB))
plt.title("Gaussian Filter 9x9")
plt.axis("off")

# Sharpening Filter
plt.subplot(4, 3, 6)
plt.imshow(cv2.cvtColor(sharpened_image_3x3, cv2.COLOR_BGR2RGB))
plt.title("Sharpening Filter 3x3")
plt.axis("off")

# Sobel Filter (Edge Detection)
plt.subplot(4, 3, 7)
plt.imshow(sobel_combined_3x3, cmap='gray')
plt.title("Sobel Edge 3x3")
plt.axis("off")

plt.subplot(4, 3, 8)
plt.imshow(sobel_combined_9x9, cmap='gray')
plt.title("Sobel Edge 9x9")
plt.axis("off")

plt.subplot(4, 3, 9)
plt.imshow(cv2.cvtColor(median_3x3, cv2.COLOR_BGR2RGB))
plt.title("Median filter 3x3")
plt.axis("off")

plt.subplot(4, 3, 10)
plt.imshow(cv2.cvtColor(median_9x9, cv2.COLOR_BGR2RGB))
plt.title("Median filter 9x9")
plt.axis("off")

plt.tight_layout()
plt.show()
