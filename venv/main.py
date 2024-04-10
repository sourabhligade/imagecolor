import cv2
import numpy as np

# Load the image
image = cv2.imread('colorful2.png')

# Reshape the image into a 2D array of pixels
pixels = image.reshape((-1, 3))


# Convert to float32
pixels = np.float32(pixels)

# Define criteria and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 5  # You can adjust this value based on your needs
_, labels, centroids = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convert centroids to integer
centroids = np.uint8(centroids)

# Find the dominant color
counts = np.bincount(labels.flatten())
dominant_color = centroids[np.argmax(counts)]

# Print the dominant color
print("Dominant color (BGR):", dominant_color)

# Display the dominant color
dominant_color_display = np.zeros((100, 100, 3), dtype=np.uint8)
dominant_color_display[:, :] = dominant_color
cv2.imshow('Dominant Color', dominant_color_display)
cv2.waitKey(0)
cv2.destroyAllWindows()
