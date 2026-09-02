import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("image.jpg",cv2.IMREAD_GRAYSCALE)

# Apply Sobel operator in x and y direction
kernel_x = np.array(
 [
  [-1, 0, 1],
  [-2, 0, 2], 
  [-1, 0, 1]
 ]
    )

# kernel_y = np.array(
#     [
#         [-1,-2,-1],
#         [0,0,0],
#         [1,2,1]
#     ]
# )
result=[]
height = image.shape[0]
length = image.shape[1]
for i in range(1,height-1):
    for j in range(1,length-1):
        part = image[i-1:i+2,j-1:j+2]
        result[i,j]=np.sum(part*kernel_x)
        

