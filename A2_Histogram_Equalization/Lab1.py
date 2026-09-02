import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("image.jpg",cv2.IMREAD_GRAYSCALE)


height = image.shape[0]
length = image.shape[1]
min_intensity = np.min(image)
max_intensity = np.max(image)

for i in range(height):
    for j in range(length):
        print(image[i][j])


plt.imshow(image,cmap= 'gray')
plt.title('Original')
plt.axis('off')
plt.show()

cv2.imwrite('image_gray.jpg',image)

for i in range(height):
    for j in range(length):
        negetive_pixel = 255 - image[i][j]
        negetive[i,j] = negetive_pixel

plt.imshow(negetive,cmap= 'gray')
plt.title('Negative')
plt.axis('off')
plt.show()

cv2.imwrite('image_negative.jpg',negetive)


for i in range(height):
    for j in range(length):
        r = int(image[i][j])
        s= c*np.log(1+r)
        log_transform[i,j] = s
        
        
output = (c*np.log(1+image,astype(np.float32))).astype(np.uint8)

for i in range(height):
    for j in range(length):
        r = int(image[i][j])/255
        s = c*(r**gamma)
        gama_image = (s*255).astype(np.uint8)


output = (c*(image/255)**gamma*255).astype(np.uint8)

plane = []
for  k in range(8):
    plane = np.empty_like(image)
    for i in range(height):
        for j in range(length):
            bit = (image[i,j] >> k) & 1
            plane[i,j] = bit*255

plane.append(plane) 



for i in range(height):
 for j in range(width):
    if (image[i, j] >= 150):
     threshold_image[i, j] = 255
    else:
     threshold_image[i, j] = 0
     
     
