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
framThresh = cv2.threshold(frameGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

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
    cnts = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    #for cnt in cnts:
    #    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt,True), True)
    #    #print(len(approx))
    #    cv2.drawContours(masked, [cnt], 0, (255, 0, 0), -1)
    
    cv2.imshow("feed", frameHSV)
    cv2.imshow("masked", masked)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
