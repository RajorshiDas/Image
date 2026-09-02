import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

input_image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

reference_image = cv2.imread("reference.jpg", cv2.IMREAD_GRAYSCALE)

input_hist = np.zeros(256)
reference_hist = np.zeros(256)
height = input_image.shape[0]
length = input_image.shape[1]
for i in range(height):
    for j in range(length):
        pixel = input_image[i][j]
        input_hist[pixel] += 1

for i in range(reference_image.shape[0]):
    for j in range(reference_image.shape[1]):
        pixel = reference_image[i][j]
        reference_hist[pixel] += 1
        

input_pdf = input_hist/(height*length)
input_cdf = np.zeros_like(input_pdf)
input_cdf[0] = input_pdf[0]
for i in range(1, 256):
    input_cdf[i] = input_cdf[i-1] + input_pdf[i]    

reference_pdf = reference_hist/(reference_image.shape[0]*reference_image.shape[1])
reference_cdf = np.zeros_like(reference_pdf)
reference_cdf[0] = reference_pdf[0]
for i in range(1, 256):
    reference_cdf[i] = reference_cdf[i-1] + reference_pdf[i]

mapping = np.zeros(256, dtype=np.uint8)
for r in range(256):
    difference = np.abs(input_cdf[r]-reference_cdf)
    z = np.argmin(difference)
    mapping[r] = z
    
output_image = mapping[input_image]
    
plt.subplot(1, 3, 1)
plt.imshow(input_image, cmap="gray")
plt.title("Input")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(reference_image, cmap="gray")
plt.title("Reference")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(output_image, cmap="gray")
plt.title("Matched")
plt.axis("off")

plt.show()