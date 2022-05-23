import os
import mediapipe as mp
import cv2
import numpy as np

def empty(tst):
            return tst

def trackbar(minval, maxval, frameSize=[520, 170]):
    frameSize[0] = 0
    frameSize[1] = 0
    frame = cv2.namedWindow("FrameSetup")
    cv2.resizeWindow("FrameSetup", frameSize[0], frameSize[1])
    cv2.createTrackbar("HUEmin", "FrameSetup", minval['HUE'][0], minval['HUE'][1], empty)
    cv2.createTrackbar("HUEmax", "FrameSetup", maxval['HUE'][0], maxval['HUE'][1], empty)
    cv2.createTrackbar("SATmin", "FrameSetup", minval['SAT'][0], minval['SAT'][1], empty)
    cv2.createTrackbar("SATmax", "FrameSetup", maxval['SAT'][0], maxval['SAT'][1], empty)
    cv2.createTrackbar("VALmin", "FrameSetup", minval['VAL'][0], minval['VAL'][1], empty)
    cv2.createTrackbar("VALmax", "FrameSetup", maxval['VAL'][0], maxval['VAL'][1], empty)


cwd = os.getcwd()
photoshot = os.path.join(cwd, 'shapes_on_canvas.jpg')
frame = cv2.imread(photoshot)

# resize frame
scale = 20
height = int(frame.shape[0] * scale / 100)
width = int(frame.shape[1] * scale / 100)
frame = cv2.resize(frame, (width, height), cv2.INTER_AREA)

# Create Trackbars -----------
minval = {"HUE":(0, 179),
          "SAT":(0, 179),
          "VAL":(0, 179)}
maxval = {"HUE":(0, 179),
          "SAT":(0, 179),
          "VAL":(0, 179)}
trackbar(minval, maxval)

#------------ Frames --------------

frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frameBlured = cv2.medianBlur(frameGray, 5)
#frameBlured = cv2.GaussianBlur(frameGray, (9, 9), 0)

cv2.imshow("frameBlured", frameBlured)
frameThresh = cv2.threshold(frameBlured, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#frameThresh = cv2.adaptiveThreshold(frameBlured, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

while True:
    # Read frames and validate reading ----------
    HUEmin = cv2.getTrackbarPos("HUEmin", "FrameSetup")
    HUEmax = cv2.getTrackbarPos("HUEmax", "FrameSetup")
    SATmin = cv2.getTrackbarPos("SATmin", "FrameSetup")
    SATmax = cv2.getTrackbarPos("SATmax", "FrameSetup")
    VALmin = cv2.getTrackbarPos("VALmin", "FrameSetup")
    VALmax = cv2.getTrackbarPos("VALmax", "FrameSetup")
    valMin = np.array([HUEmin, SATmin, VALmin])
    valMax = np.array([HUEmax, SATmax, VALmax])

    # Implement values on the masked frame -----------------
    masked = cv2.inRange(frameHSV, valMin, valMax)
    cnts, hierarchy = cv2.findContours(frameThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area > 500:
            approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt,True), True)
            cv2.drawContours(frame, [cnt], -1, (255, 0, 0), 2)
            print(approx)
            print("---------------------------------------------------------------")
    
    cv2.imshow("frameThresh", frameThresh)
    cv2.imshow("feed", frame)
    #cv2.imshow("frameBlured", frameBlured)
    
    #cv2.imshow("masked", masked)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
