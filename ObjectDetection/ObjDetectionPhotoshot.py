import os
import mediapipe as mp
import cv2
import numpy as np
from pathlib import Path


def detectShape(contour):
    shape = "unidentified"
    arcLength = cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour, 0.01*arcLength, True)    
    print(len(approx))
    if len(approx) == 3:
        shape = "triangle"
    elif len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(contour)
        ratio = w / h
        shape = "square" if ratio > 0.95 and ratio < 1.05 else "rectangle"
    elif len(approx) == 5:
        shape = "pentagon"
    elif len(approx) == 6:
        shape = "hexagon"
    elif len(approx) == 8:
        shape = "screw"
    elif len(approx) > 8:
        shape = "circle"
    #cv2.rectangle(frame, x, y, (150,0,150), 2)
    return shape

try:
    BASE_DIR = Path(__file__).resolve().parent
    photoshot = os.path.join(BASE_DIR, 'shapes/rectangle.jpg')
    print(photoshot)
    frame = cv2.imread(photoshot)
    scale = 80
    height = int(frame.shape[0] * scale / 100)
    width = int(frame.shape[1] * scale / 100)
    frame = cv2.resize(frame, (width, height), cv2.INTER_AREA)
except:
    print("Can't open file.")

#------------ Frames --------------
#               Blurring
#frameBlured = cv2.medianBlur(frameGray, 5)
frameBlured = cv2.GaussianBlur(frame, (17, 17), 0)
cv2.imshow("frameBlured", frameBlured)

frameGray = cv2.cvtColor(frameBlured, cv2.COLOR_BGR2GRAY)

#               Threshing
#frameThresh = cv2.threshold(frameBlured, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
frameThresh = cv2.threshold(frameGray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
#frameThresh = cv2.adaptiveThreshold(frameBlured, 255,
#                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                    cv2.THRESH_BINARY, 11, 2)

cannyT1 = 20
cannyT2 = 20
frameThresh = cv2.Canny(frameThresh, cannyT1, cannyT2)

kernel = np.ones((5, 5))
frameThresh = cv2.dilate(frameThresh, kernel, iterations=2)

while True:   

    
    contours, hierarchy = cv2.findContours( frameThresh,
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE  )
    #trackbar(minval, maxval)
    for contour in contours:
        #print(contour)
        cv2.drawContours(frame, contours, 0, (50, 150, 150), 3)
        area = cv2.contourArea(contour)
        if area > 1500:
            shape = detectShape(contour)
            print(shape)
            break
            #momment = cv2.moments(contour)
            #print(momment)
            

            
    
    cv2.imshow("frameThresh", frameThresh)
    cv2.imshow("feed", frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
