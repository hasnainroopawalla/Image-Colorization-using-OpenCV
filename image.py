import numpy as np
import process
import cv2

input_img = "images/b24.jpg"

image = cv2.imread(input_img)
colorized = process.processimg(image)

cv2.imshow("Original", image)
cv2.imshow("Colored", colorized)
cv2.waitKey(0)
