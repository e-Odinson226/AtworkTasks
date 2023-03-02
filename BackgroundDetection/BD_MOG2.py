import numpy as np
import cv2


# frame = cv2.imread("../test_images/Axis_4.jpg")
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("video.avi")
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)

    cv2.imshow("frame", fgmask)

    if cv2.waitKey(30) & 0xFF == 27:
        break


cv2.destroyAllWindows()