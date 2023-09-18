import time
import cv2
import numpy as np

# Read image
img = cv2.imread('image.jpg')

# Making kernel
kernel = np.array([[1, 1, 1],
                   [1,-8, 1],
                   [1, 1, 1]])
kernel = kernel/(np.sum(kernel) if np.sum(kernel)!=0 else 1)

# Start time
start = time.time()

# Apply convolution
imgResult = cv2.filter2D(img,-1,kernel)

# End time
end = time.time()

print("Execution time: ",(end-start), "second")

# Show images
cv2.imshow('Citra Asli', img)
cv2.imshow('Hasil Convolosi', imgResult)

cv2.waitKey(0)
cv2.destroyAllWindows()