from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_PATH = SCRIPT_DIR / "image.jpg"

image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

size = 5
sigma = 1

# Create a 2D Gaussian kernel
kernel = np.zeros((size,size))

center = size//2

for  i in range(size):
    for j in range(size):
        x = i - center
        y = j - center
        
        # kernel[i,j]= np.exp(-(x**2+y**2)/(2*sigma**2))
        # kernel[i,j]= (1/(2*np.pi*sigma**2))*np.exp(-(x**2+y**2)/(2*sigma**2))
        # kernel[i,j]= (-x/(sigma*sigma))*np.exp(-(x**2+y**2)/(2*sigma**2))
        r = x**2+y**2
        kernel[i,j]= ((r-2*sigma**2)/(sigma**4))*np.exp(-r/(2*sigma**2))
        
        
        
       
kernel = kernel / np.sum(kernel)
print(kernel)

height = image.shape[0]
length = image.shape[1]

result = np.zeros((height,length))

pad = size //2

for i in range(pad,height-pad):
    for j in range(pad,length-pad):
        part= image[i-pad:i+pad+1, j-pad:j+pad+1]
        
        result[i,j]= np.sum(part*kernel)
        
result = np.clip(result,0,255).astype(np.uint8)

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(result, cmap="gray")
plt.title("Filtered")
plt.axis("off")

plt.show()