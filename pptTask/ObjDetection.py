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

while(True):
    # Read frames and validate reading ----------    
    succ, frame = cap.read()
    if not succ:
        print("failed to read input frames")
        break
    
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow("feed", frameHSV)
    
    # Read frames and validate reading ----------
    HUEmin = cv2.getTrackbarPos("HUEmin", "FrameSetup")
    HUEmax = cv2.getTrackbarPos("HUEmax", "FrameSetup")
    SATmin = cv2.getTrackbarPos("SATmin", "FrameSetup")
    SATmax = cv2.getTrackbarPos("SATmax", "FrameSetup")
    VALmin = cv2.getTrackbarPos("VALmin", "FrameSetup")
    VALmax = cv2.getTrackbarPos("VALmax", "FrameSetup")
    valMin = np.array([HUEmin, SATmin, VALmin])
    valMax = np.array([HUEmax, SATmax, VALmax])
    print(f'values min: {valMin}')
    print(f'values max: {valMax}')
    
    # Implement values on the masked frame -----------------
    masked = cv2.inRange(frameHSV, valMin, valMax )
    cv2.imshow("masked", masked)
    
    #height, width, fra = frame.shape
    blankImage = np.zeros(frame.shape, np.uint8)

    contours = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #contours = contours[0] if len(contours) == 2 else contours[1]
    for contour in contours:
        area = cv2.contourArea(contour)
        approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
        print(len(approx))
        if len(approx)==5:
            print("Blue = pentagon")
            cv2.drawContours(blankImage,[contour],0,255,-1)
        elif len(approx)==3:
            print("Green = triangle")
            cv2.drawContours(blankImage,[contour],0,(0,255,0),-1)
        elif len(approx)==4:
            print("Red = square")
            cv2.drawContours(blankImage,[contour],0,(0,0,255),-1)
        elif len(approx) == 6:
            print("Cyan = Hexa")
            cv2.drawContours(blankImage,[contour],0,(255,255,0),-1)
        elif len(approx) == 8:
            print("White = Octa")
            cv2.drawContours(blankImage,[contour],0,(255,255,255),-1)
        elif len(approx) > 12:
            print("Yellow = circle")
            cv2.drawContours(blankImage,[contour],0,(0,255,255),-1)
    
    
    cv2.imshow("Drawed shapes", blankImage)
    
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
    