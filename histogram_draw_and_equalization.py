# Calculate and plot the histogram for both X1 and X2
# And plot the histogram after equalizing X2
import numpy as np
import matplotlib.pyplot as plt
import cv2

matrix_x1 = cv2.imread("Image_for_TeamSV/bagues.jpg")

matrix_x2 = cv2.imread("Image_for_TeamSV/bagues.jpg")

hist, bins = np.histogram(matrix_x2.flatten(), bins=256, range=(0, 255))
cdf = hist.cumsum()  # cumulative distribution function
cdf_normalized = cdf * (255 / cdf[-1])  # normalize to 0-255

# Step 2: Use CDF to map original intensities to equalized intensities
equalized_matrix_x2 = cdf_normalized[matrix_x2].astype(int)


# Histogram for X1
hist_x1, bins_x1 = np.histogram(matrix_x1.flatten() / 2, bins=range(256))
# Histogram for X2
hist_x2, bins_x2 = np.histogram(matrix_x2.flatten(), bins=range(256))
# Histogram for equalized X2
hist_equalized_x2, bins_equalized_x2 = np.histogram(equalized_matrix_x2.flatten(), bins=range(257))

# Plotting
plt.figure(figsize=(12, 6))

# Histogram of X1
plt.subplot(1, 3, 1)
plt.bar(bins_x1[:-1], hist_x1, width=1, edgecolor="black")
plt.title("Histogram of X1")
plt.xlabel("Intensity Value")
plt.ylabel("Frequency")

# Histogram of X2
plt.subplot(1, 3, 2)
plt.bar(bins_x2[:-1], hist_x2, width=1, edgecolor="black")
plt.title("Histogram of X2")
plt.xlabel("Intensity Value")

# Histogram of Equalized X2
plt.subplot(1, 3, 3)
plt.bar(bins_equalized_x2[:-1], hist_equalized_x2, width=1, edgecolor="black")
plt.title("Histogram of Equalized X2")
plt.xlabel("Intensity Value")

plt.tight_layout()
plt.show()
