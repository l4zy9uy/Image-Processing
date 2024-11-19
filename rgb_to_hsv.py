# Load the new color image
import cv2
import numpy as np
from PIL import ImageOps
from PIL.Image import Image
from matplotlib import pyplot as plt

# Load the color image in OpenCV format
image_cv_color = cv2.imread("Image_for_TeamSV/car.jpg")  # OpenCV reads in BGR format

# 1. Convert RGB (BGR in OpenCV) to HSV using OpenCV
hsv_image_cv = cv2.cvtColor(image_cv_color, cv2.COLOR_BGR2HSV)

# 2. Negative (Inversion) Transformation using OpenCV
negative_image_cv = cv2.bitwise_not(image_cv_color)

# Convert images back to RGB format for display with matplotlib
#hsv_image_cv_rgb = cv2.cvtColor(hsv_image_cv, cv2.COLOR_HSV2RGB)
negative_image_cv_rgb = cv2.cvtColor(negative_image_cv, cv2.COLOR_BGR2RGB)
original_image_rgb = cv2.cvtColor(image_cv_color, cv2.COLOR_BGR2RGB)

# Display the results using matplotlib
plt.figure(figsize=(12, 4))

# Original Image
plt.subplot(1, 3, 1)
plt.imshow(original_image_rgb)
plt.title("Original Image")
plt.axis("off")

# HSV Image
plt.subplot(1, 3, 2)
plt.imshow(hsv_image_cv)
plt.title("HSV Color Space (OpenCV)")
plt.axis("off")

# Negative Image
plt.subplot(1, 3, 3)
plt.imshow(negative_image_cv_rgb)
plt.title("Negative Image (OpenCV)")
plt.axis("off")

plt.show()
