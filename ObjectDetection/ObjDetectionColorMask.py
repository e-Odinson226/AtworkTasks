from re import U
from xml.dom import HierarchyRequestErr
import cv2
import time
import numpy as np

def empty(tst):
        pass

def trackbar(modes = [0, 1, 2, 3], frameSize=[520, 170]):
    frame = cv2.namedWindow("Detection Mode")
    cv2.resizeWindow("Detail tweaks", frameSize[0], frameSize[1])
    cv2.createTrackbar("mode", "Detection Mode", modes[0], modes[3], empty)
    cv2.createTrackbar("thresh", "Detection Mode", 0, 255, empty)

trackbar()


try:
    cap = cv2.VideoCapture(0)
    cap.set(3, 1080)
    cap.set(4, 720)    
except:
    print("Can't read video frame.")

while True:
    mode = cv2.getTrackbarPos("mode", "Detail tweaks")
    threshValue = cv2.getTrackbarPos("thresh", "Detail tweaks")
    
    
    
    isReadOk, frame = cap.read()
    frame = cv2.flip(frame, 1)
    begin = time.time()
    
    # -----------------------------
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameBlured = cv2.GaussianBlur(frameGray, (3, 3), 0)
    
    if mode == 0:
        frameThresh = cv2.threshold(frameBlured, threshValue, 255,
                                cv2.THRESH_OTSU)[1]
        
    elif mode == 1:
        loweBound = np.array([100, 200, 300])
        loweBound = np.array([100, 200, 300])
        frameThresh = cv2.threshold(frameBlured, threshValue, 255,
                                cv2.THRESH_BINARY_INV)[1]
        frameThresh = ~frameThresh
        
    elif mode == 2:
        HUEmin = cv2.getTrackbarPos("HUEmin", "FrameSetup")
        HUEmax = cv2.getTrackbarPos("HUEmax", "FrameSetup")
        SATmin = cv2.getTrackbarPos("SATmin", "FrameSetup")
        SATmax = cv2.getTrackbarPos("SATmax", "FrameSetup")
        VALmin = cv2.getTrackbarPos("VALmin", "FrameSetup")
        VALmax = cv2.getTrackbarPos("VALmax", "FrameSetup")
        lowerBound = np.array([HUEmin, SATmin, VALmin])
        upperBound = np.array([HUEmax, SATmax, VALmax])
        frameThresh = cv2.inRange(frame, lowerBound, upperBound)
    
    
    contours, hierarchy = cv2.findContours(frameThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cont in contours:
        cv2.drawContours(frame, cont, -1, (255, 0, 255), 2)
        (x,y,w,h) = cv2.boundingRect(cont)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (10,0,10), 2)
    # -----------------------------
    
    end = time.time()
    fps = 1 /(end-begin)
    cv2.putText(frame, f"fps:{int(fps)}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (20,20,20), 2)
    
    cv2.imshow("frameThresh", frameThresh)
    cv2.imshow("feed", frame)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break