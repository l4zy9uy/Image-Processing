# Reapply the filters with boundary set to 0 (padding with constant values)
from scipy.ndimage import convolve, median_filter, minimum_filter, maximum_filter
import numpy as np

X2_matrix = np.array([
    [19, 17, 2, 1, 1, 2, 2, 1],
    [18, 19, 19, 17, 1, 1, 1, 1],
    [17, 18, 19, 17, 1, 2, 1, 1],
    [18, 19, 19, 19, 19, 1, 2, 3],
    [18, 19, 19, 18, 18, 17, 2, 3],
    [19, 19, 18, 18, 18, 2, 2, 1],
    [19, 19, 19, 18, 17, 1, 2, 1],
    [18, 19, 18, 17, 3, 1, 1, 3]
], dtype=np.uint8)

mean_filter = np.ones((3, 3)) / 9
gaussian_filter = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
sharpening_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
laplacian_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

# Linear Filters
mean_filtered_zero = convolve(X2_matrix, mean_filter, mode='constant', cval=0)
gaussian_filtered_zero = convolve(X2_matrix, gaussian_filter, mode='constant', cval=0)
sharpened_filtered_zero = convolve(X2_matrix, sharpening_filter, mode='constant', cval=0)
sobel_filtered_zero = convolve(X2_matrix, sobel_x, mode='constant', cval=0)
laplacian_filtered_zero = convolve(X2_matrix, laplacian_filter, mode='constant', cval=0)

# Non-Linear Filters with zero padding
median_filtered_zero = median_filter(X2_matrix, size=3, mode='constant', cval=0)
min_filtered_zero = minimum_filter(X2_matrix, size=3, mode='constant', cval=0)
max_filtered_zero = maximum_filter(X2_matrix, size=3, mode='constant', cval=0)

# Organize results with zero padding into DataFrames for display
filtered_results_zero_padding = {
    "Mean Filter (Zero Boundary)": mean_filtered_zero,
    "Gaussian Filter (Zero Boundary)": gaussian_filtered_zero,
    "Sharpening Filter (Zero Boundary)": sharpened_filtered_zero,
    "Sobel Filter (Zero Boundary)": sobel_filtered_zero,
    "Laplacian Filter (Zero Boundary)": laplacian_filtered_zero,
    "Median Filter (Zero Boundary)": median_filtered_zero,
    "Min Filter (Zero Boundary)": min_filtered_zero,
    "Max Filter (Zero Boundary)": max_filtered_zero
}

for key, value in filtered_results_zero_padding.items():
    print(f"{key}:\n{value}\n")
