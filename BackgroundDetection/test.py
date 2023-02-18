import numpy as np
import cv2


frame = cv2.imread("../test_images/Axis_4.jpg")
cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()

while 1:
    re, frame = cap.read()
    fgmask = fgbg.apply(frame)

    cv2.imshow("frame", fgmask)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break


cv2.destroyAllWindows()
