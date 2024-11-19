import numpy as np
import matplotlib.pyplot as plt

# Original 8x8 matrix (X1 in the image)
matrix = np.array([
    [19, 17, 2, 1, 1, 2, 2, 1],
    [18, 19, 19, 17, 1, 1, 1, 1],
    [17, 18, 19, 17, 1, 2, 1, 1],
    [18, 19, 19, 19, 19, 1, 2, 3],
    [18, 19, 19, 18, 18, 17, 2, 3],
    [19, 19, 18, 18, 18, 2, 2, 1],
    [19, 19, 19, 18, 17, 1, 2, 1],
    [18, 19, 18, 17, 3, 1, 1, 3]
])

# Stretching parameters
I_min, I_max = 0, 255
S_min, S_max = matrix.min(), matrix.max()

print(S_min, S_max)

# Histogram stretching
stretched_matrix = ((I_max - I_min) / (S_max - S_min)) * matrix + (I_min * S_max - I_max * S_min) / (S_max - S_min)
stretched_matrix = np.clip(stretched_matrix, I_min, I_max).astype(int)

# Histogram equalization
# Step 1: Calculate histogram and CDF
hist, bins = np.histogram(matrix.flatten(), bins=256, range=(I_min, I_max))
cdf = hist.cumsum()  # cumulative distribution function
cdf_normalized = cdf * (255 / cdf[-1])  # normalize to 0-255

# Step 2: Use CDF to map original intensities to equalized intensities
equalized_matrix = cdf_normalized[matrix].astype(int)

print(stretched_matrix)

print(equalized_matrix)