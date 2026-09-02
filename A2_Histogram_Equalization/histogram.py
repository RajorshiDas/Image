import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
hist = np.zeros(256)
height = image.shape[0]
length = image.shape[1]

for i in range(height):
    for j in range(length):
        pixel = image[i][j]
        hist[pixel] += 1

pdf =  hist/(height*length)

cdf = np.cumsum(pdf)

mapping = np.round(255*cdf).astype(np.uint8)

result = mapping[image]
    

plt.plot(hist)
plt.title("Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.show()

