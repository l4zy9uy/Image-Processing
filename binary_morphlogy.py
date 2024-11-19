from skimage.morphology import binary_erosion, binary_dilation, binary_opening, binary_closing
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

threshold = (X2_matrix.max() + X2_matrix.min()) / 2
binary_X2 = (X2_matrix > threshold).astype(int)

structuring_element = np.ones((3, 3), dtype=int)

# Re-apply morphological operations with correct parameter
binary_erosion_X2 = binary_erosion(binary_X2, footprint=structuring_element).astype(int)
binary_dilation_X2 = binary_dilation(binary_X2, footprint=structuring_element).astype(int)
binary_opening_X2 = binary_opening(binary_X2, footprint=structuring_element).astype(int)
binary_closing_X2 = binary_closing(binary_X2, footprint=structuring_element).astype(int)

# Organize results for display
binary_morphology_results = {
    "Binary Image (Thresholded)": binary_X2,
    "Binary Erosion": binary_erosion_X2,
    "Binary Dilation": binary_dilation_X2,
    "Binary Opening": binary_opening_X2,
    "Binary Closing": binary_closing_X2
}

for key, value in binary_morphology_results.items():
    print(f"{key}:\n{value}\n")
