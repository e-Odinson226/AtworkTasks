import os
import mediapipe as mp
import cv2
import numpy as np
from matplotlib import pyplot as plt


try:
    cwd = os.getcwd()
    photoshot = os.path.join(cwd, 'shapes_on_canvas.jpg')
    frame = cv2.imread(photoshot)
except:
    print("Can't open file.")

# resize frame
scale = 20
height = int(frame.shape[0] * scale / 100)
width = int(frame.shape[1] * scale / 100)
frame = cv2.resize(frame, (width, height), cv2.INTER_AREA)


#------------ Frames --------------
frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# Blurring
#frameBlured = cv2.medianBlur(frameGray, 5)
#frameBlured = cv2.GaussianBlur(frameGray, (9, 9), 0)

# Threshing
#frameThresh = cv2.threshold(frameBlured, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
frameThresh = cv2.threshold(frameGray, 170, 255, cv2.THRESH_BINARY_INV)[1]
#frameThresh = cv2.adaptiveThreshold(frameBlured, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

while True:
    cnts, hierarchy = cv2.findContours( frameThresh,
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE  )
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area > 500:
            arcLength = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt, 0.01*arcLength, True)
            cv2.drawContours(frame, [cnt], 0, (255, 0, 0), 2)
            #print(approx)
            #print("---------------------------------------------------------------")
    
    cv2.imshow("frameThresh", frameThresh)
    cv2.imshow("feed", frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
