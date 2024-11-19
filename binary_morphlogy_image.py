import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.morphology import binary_erosion, binary_dilation, binary_opening, binary_closing, square

# Define a cross-shaped mask manually
cross_mask = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
], dtype=bool)

image = cv2.imread("Image_for_TeamSV/car.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

threshold = (gray_image.max() + gray_image.min()) / 2
binary_image = (gray_image  > threshold).astype(int)


# Apply morphological operations with square mask
erosion_square = binary_erosion(binary_image, cross_mask)
dilation_square = binary_dilation(binary_image, cross_mask)
opening_square = binary_opening(binary_image, cross_mask)
closing_square = binary_closing(binary_image, cross_mask)

# Apply morphological operations with cross mask
erosion_cross = binary_erosion(binary_image, cross_mask)
dilation_cross = binary_dilation(binary_image, cross_mask)
opening_cross = binary_opening(binary_image, cross_mask)
closing_cross = binary_closing(binary_image, cross_mask)

# Display results using matplotlib
plt.figure(figsize=(12, 8))

# Original Binary Image
plt.subplot(4, 5, 1)
plt.imshow(binary_image, cmap='gray')
plt.title("Original Image")
plt.axis("off")

# Square Mask Results
plt.subplot(4, 5, 2)
plt.imshow(erosion_square, cmap='gray')
plt.title("Erosion (Square)")
plt.axis("off")

plt.subplot(4, 5, 3)
plt.imshow(dilation_square, cmap='gray')
plt.title("Dilation (Square)")
plt.axis("off")

plt.subplot(4, 5, 4)
plt.imshow(opening_square, cmap='gray')
plt.title("Opening (Square)")
plt.axis("off")

plt.subplot(4, 5, 5)
plt.imshow(closing_square, cmap='gray')
plt.title("Closing (Square)")
plt.axis("off")

# Cross Mask Results
plt.subplot(4, 5, 7)
plt.imshow(erosion_cross, cmap='gray')
plt.title("Erosion (Cross)")
plt.axis("off")

plt.subplot(4, 5, 8)
plt.imshow(dilation_cross, cmap='gray')
plt.title("Dilation (Cross)")
plt.axis("off")

plt.subplot(4, 5, 9)
plt.imshow(opening_cross, cmap='gray')
plt.title("Opening (Cross)")
plt.axis("off")

plt.subplot(4, 5, 10)
plt.imshow(closing_cross, cmap='gray')
plt.title("Closing (Cross)")
plt.axis("off")

plt.tight_layout()
plt.show()
