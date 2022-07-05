from matplotlib import pyplot as plt
import time
import cv2

def empty(tst):
        pass
    
def trackbar(modes = [0, 1], frameSize=[520, 170]):
    frame = cv2.namedWindow("Tweaks")
    cv2.resizeWindow("Tweaks", frameSize[0], frameSize[1])
    
    cv2.createTrackbar("mode", "Tweaks", modes[0], modes[1], empty)
    cv2.createTrackbar("thresh", "Tweaks", 0, 255, empty)

#try:
#    cap = cv2.VideoCapture(2)
#    
#    cap.set(3, 720)
#    cap.set(4, 480)    
#except:
#    print("Can't read video frame.")

#cv2.namedWindow("Tweaks")
#cv2.resizeWindow("Tweaks", 300, 250)
#trackbar()

while True:
    #mode = cv2.getTrackbarPos('mode', 'Tweaks')
    #threshValue = cv2.getTrackbarPos('thresh', 'Tweaks')
#---------- BEGINING TO READ
    #_, frame = cap.read()
    frame = cv2.imread('./img samples/IMG_20220705_165233.jpg')
    frame = cv2.resize(frame, (720, 480), interpolation= INTER_AREA)
    cv2.imshow('frame', frame)
    #frame = cv2.flip(frame, 1)
#---------- BEGINING TO DO COMPUTING ON FRAMES
    begin = time.time()
    
    #---------- CHOOSE THRESHOLD METHOD
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameBlured = cv2.GaussianBlur(frameGray, (11, 11), 0)
    cv2.imshow("frameBL", frameBlured)
    frameDenoised = cv2.fastNlMeansDenoising(frame, None, 10, 10, 7, 21)
    cv2.imshow("frameDN", frameDenoised)
    
    
    
    #if mode == 0:
    #    frameThresh = cv2.threshold(frameBlured, 100, 255,
    #                            cv2.THRESH_OTSU)[1]
    #elif mode == 1:
    #    frameThresh = cv2.threshold(frameBlured, threshValue, 255,
    #                            cv2.THRESH_BINARY_INV)[1]
    #    frameThresh = ~frameThresh
    #----------------------------------
    #---------- DRAW CONTOUR AROUND OBJECTS
    #contours, hierarchy = cv2.findContours(frameThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #for cnt in contours:
    #    cv2.drawContours(frame, cnt, -1, (255, 0, 255), 2)
    #    (x,y,w,h) = cv2.boundingRect(cnt)
    #    cv2.rectangle(frame, (x, y), (x+w, y+h), (10,0,10), 2)

    
    
    #---------- END OF FRAME COMPUTING PART
    end = time.time()
    fps = 1 /(end-begin)
    cv2.putText(frame, f"fps:{int(fps)}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (20,20,20), 2)
    
    #cv2.imshow("frameThresh", frameThresh)
    cv2.imshow("feed", frame)
    #cv2.imshow("frameThresh", frameThresh)
        
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break