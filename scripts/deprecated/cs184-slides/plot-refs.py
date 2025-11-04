

import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np

# Load the .mat file
data = loadmat('./data/ref_cyflower1bb_reg1.mat')

# Assuming the hyperspectral image is stored in a variable named 'image' in the .mat file
# Replace 'image' with the actual key in the .mat file if different
hyperspectral_image = data['reflectances']
# Display the first band of the hyperspectral image for interactive selection
image = plt.imread('./data/cyflower1bb_reg1_2bright.bmp')
plt.imshow(image)
plt.title("Select a spatial location")
plt.xlabel("Column")
plt.ylabel("Row")

# Use ginput to select a point interactively
# Enable interactive mode for matplotlib

# Use ginput to select a point interactively
selected_point = plt.ginput(1)
plt.close()

# Extract the row and column from the selected point
row, col = int(selected_point[0][1]), int(selected_point[0][0])
print(f"Selected point: ({row}, {col})")


# Extract the reflectance spectrum at the selected location
reflectance_spectrum = hyperspectral_image[row, col, :]

# Plot the reflectance spectrum
plt.figure(figsize=(10, 6))
plt.plot(reflectance_spectrum)
plt.title(f"Reflectance Spectrum at Location ({row}, {col})")
plt.xlabel("Wavelength Index")
plt.ylabel("Reflectance")
plt.grid(True)
plt.show()
