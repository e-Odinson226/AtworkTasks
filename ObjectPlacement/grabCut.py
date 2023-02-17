from matplotlib import pyplot as plt
import numpy as np
import cv2


original = cv2.imread("../ImageDataset_makan_100/Color/2.jpg")
img = original.copy()
mask = np.zeros(img.shape[0:2], np.uint8)

cv2.imshow("mask", img)
cv2.waitKey()
