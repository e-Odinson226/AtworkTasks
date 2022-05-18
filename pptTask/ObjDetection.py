from inspect import getcallargs
from itertools import count
from matplotlib.pyplot import get
import mediapipe as mp
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Create Trackbars -----------
def empty(tst):
    return tst
cv2.namedWindow("FrameSetup")
cv2.resizeWindow("FrameSetup", 520, 170)
cv2.createTrackbar("HUEmin", "FrameSetup", 44, 179, empty)
cv2.createTrackbar("HUEmax", "FrameSetup", 179, 179, empty)
cv2.createTrackbar("SATmin", "FrameSetup", 0, 255, empty)
cv2.createTrackbar("SATmax", "FrameSetup", 255, 255, empty)
cv2.createTrackbar("VALmin", "FrameSetup", 0, 255, empty)
cv2.createTrackbar("VALmax", "FrameSetup", 93, 255, empty)

def showContours(img, frameContour):
    contours, heirarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04*peri , True)
            #cv2.drawContours(frameContour, [contour], -1, (255, 255, 0),3 )
            print(f'length approx:{len(approx)} | area:{area}')
            if len(approx)==5:
                print("Blue = pentagon")
                cv2.drawContours(frameContour,[contour],0,255,-1)
            elif len(approx)==3:
                print("Green = triangle")
                cv2.drawContours(frameContour,[contour],0,(0,255,0),-1)
            elif len(approx)==4:
                print("Red = square")
                cv2.drawContours(frameContour,[contour],0,(0,0,255),-1)
            elif len(approx) == 6:
                print("Cyan = Hexa")
                cv2.drawContours(frameContour,[contour],0,(255,255,0),-1)
            elif len(approx) == 8:
                print("White = Octa")
                cv2.drawContours(frameContour,[contour],0,(255,255,255),-1)
            elif len(approx) > 12:
                print("Yellow = circle")
                cv2.drawContours(frameContour,[contour],0,(0,255,255),-1)
    

while(True):
    # Read frames and validate reading ----------    
    succ, frame = cap.read()
    if not succ:
        print("failed to read input frames")
        break
    
    maskedFrameContour = frame.copy()
    threshFrameContour = frame.copy()
    # Create and represent hsv version of input frames ---------
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #cv2.imshow("feed", frameHSV)
    
    # Create and represent canny version of input frames ---------
    frameCanny = cv2.Canny(cv2.GaussianBlur(frame,(25, 25), 5), 7, 7)
    #cv2.imshow("Canny", frameCanny)
    
    # Create and represent threshed version of input frames ---------
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Gray", frameGray)
    
    # Create and represent thresh version of input frames ---------
    frameThresh = cv2.threshold(frameGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU )[1]
    cv2.imshow("Thresh", frameThresh)
    
    
    # Read frames and validate reading ----------
    HUEmin = cv2.getTrackbarPos("HUEmin", "FrameSetup")
    HUEmax = cv2.getTrackbarPos("HUEmax", "FrameSetup")
    SATmin = cv2.getTrackbarPos("SATmin", "FrameSetup")
    SATmax = cv2.getTrackbarPos("SATmax", "FrameSetup")
    VALmin = cv2.getTrackbarPos("VALmin", "FrameSetup")
    VALmax = cv2.getTrackbarPos("VALmax", "FrameSetup")
    valMin = np.array([HUEmin, SATmin, VALmin])
    valMax = np.array([HUEmax, SATmax, VALmax])
    #print(f'values min: {valMin}')
    #print(f'values max: {valMax}')
    
    # Implement values on the masked frame -----------------
    frameMasked = cv2.inRange(frameHSV, valMin, valMax )
    cv2.imshow("masked", frameMasked)
    
    
    #height, width, fra = frame.shape
    #blankImage = np.zeros(frame.shape, np.uint8)
    showContours(frameMasked, maskedFrameContour)
    cv2.imshow('maskedFrameContour', maskedFrameContour)
    showContours(frameThresh, threshFrameContour)
    cv2.imshow('threshFrameContour', threshFrameContour)
    #showContours(blankImage, 'blankImage')
    #showContours(frameThresh)
    
    # Create and represent thresh version of input frames ---------



    
    
    #approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)

    #contours = contours[0] if len(contours) == 2 else contours[1]
    #approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
    