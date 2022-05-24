import os
import mediapipe as mp
import cv2
import numpy as np
from pathlib import Path

def empty(tst):
        pass
    
def trackbar(minval, maxval, frameSize=[520, 170]):
    frameSize[0] = 0
    frameSize[1] = 0
    frame = cv2.namedWindow("FrameSetup")
    cv2.resizeWindow("FrameSetup", frameSize[0], frameSize[1])
    cv2.createTrackbar("Canny t1", "FrameSetup", minval['canny'][0], minval['canny'][1], empty)
    cv2.createTrackbar("Canny t2", "FrameSetup", maxval['canny'][0], maxval['canny'][1], empty)

# Create Trackbars -----------
minval = {"canny":(86, 254)}
maxval = {"canny":(203, 254)}
trackbar(minval, maxval)

def detectShape(contour):
    shape = "unidentified"
    arcLength = cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour, 0.01*arcLength, True)
    cv2.drawContours(frame, [contour], 0, (20, 250, 200), 3)
    (x, y, w, h) = cv2.boundingRect(contour)
    print(len(approx))
    if len(approx) == 3:
        shape = "triangle"
    elif len(approx) == 4:
        
        ratio = w / h
        shape = "square" if ratio > 0.95 and ratio < 1.05 else "rectangle"
    elif len(approx) == 5:
        
        shape = "pentagon"
    elif len(approx) == 6:
        
        shape = "hexagon"
    elif len(approx) == 8:
        
        shape = "screw"
    elif len(approx) == 20:
        
        shape = "Bracket"
    cv2.rectangle(frame, (x, y), (x + w, y + h), (150,0,150), 2)
    cv2.putText(frame,
                f"corners: {len(approx)}",
                (x , y-30),
                cv2.FONT_HERSHEY_COMPLEX, 1,
                (150,0,150), 2)
    
    cv2.putText(frame,
                f"shape:{shape}",
                (x , y-10),
                cv2.FONT_HERSHEY_COMPLEX, 1,
                (150,0,150), 2)
    return shape

try:
    BASE_DIR = Path(__file__).resolve().parent
    photoshot = os.path.join(BASE_DIR, 'shapes/shapes_on_canvas_croped.jpg')
    print(photoshot)
    frame = cv2.imread(photoshot)
    scale = 20
    height = int(frame.shape[0] * scale / 100)
    width = int(frame.shape[1] * scale / 100)
    frame = cv2.resize(frame, (width, height), cv2.INTER_AREA)
except:
    print("Can't open file.")

#------------ Frames --------------
#               Blurring
#frameBlured = cv2.medianBlur(frameGray, 5)
frameBlured = cv2.GaussianBlur(frame, (17, 17), 0)

frameGray = cv2.cvtColor(frameBlured, cv2.COLOR_BGR2GRAY)

#               Threshing
#frameThresh = cv2.threshold(frameBlured, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#frameThresh = cv2.threshold(frameGray, threshT1, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
#frameThresh = cv2.adaptiveThreshold(frameBlured, 255,
#                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                    cv2.THRESH_BINARY, 11, 2)
kernel = np.ones((5, 5))

while True:
    cannyT1 = cv2.getTrackbarPos("Canny t1", "FrameSetup")
    cannyT2 = cv2.getTrackbarPos("Canny t2", "FrameSetup")
        
    frameThresh = cv2.Canny(frameGray, cannyT1, cannyT2)
    cv2.imshow("Canny", frameThresh)
    frameDilate = cv2.dilate(frameThresh, kernel, iterations=1)
    #cv2.imshow("Dilated", frameDilate)
    

    
    contours, hierarchy = cv2.findContours( frameThresh,
                                        cv2.RETR_LIST,
                                        cv2.CHAIN_APPROX_SIMPLE  )
    #trackbar(minval, maxval)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 200:
            shape = detectShape(contour)
            print(shape)
        
            

            
    
    cv2.imshow("frameThresh", frameThresh)
    cv2.imshow("feed", frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
