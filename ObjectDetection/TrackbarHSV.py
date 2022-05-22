import cv2
class Trackbar:
    def empty(self, tst):
            return tst
    def __init__(self, minval, maxval):
        cv2.namedWindow("FrameSetup")
        cv2.resizeWindow("FrameSetup", 520, 170)
        cv2.createTrackbar("HUEmin", "FrameSetup", minval['HUE'][0], minval['HUE'][1], self.empty)
        cv2.createTrackbar("HUEmax", "FrameSetup", maxval['HUE'][0], maxval['HUE'][1], self.empty)
        cv2.createTrackbar("SATmin", "FrameSetup", minval['SAT'][0], minval['SAT'][1], self.empty)
        cv2.createTrackbar("SATmax", "FrameSetup", maxval['SAT'][0], maxval['SAT'][1], self.empty)
        cv2.createTrackbar("VALmin", "FrameSetup", minval['VAL'][0], minval['VAL'][1], self.empty)
        cv2.createTrackbar("VALmax", "FrameSetup", maxval['VAL'][0], maxval['VAL'][1], self.empty)
        
    
