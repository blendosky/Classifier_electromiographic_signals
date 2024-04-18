import cv2
import numpy as np


float_img = np.random.random((280,280))
im = np.array(float_img * 255, dtype = np.uint8)
threshed = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)

#Displayed the image
cv2.imshow("WindowNameHere", threshed)
cv2.waitKey(0)