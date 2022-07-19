from matplotlib import pyplot as plt
import time
import cv2
import numpy as np

def preprocess_frame(frame):
    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #return cv2.resize(frame, (1920, 1080), interpolation= cv2.INTER_AREA)
    return frame

def median_adaptive_th(frame):
    #blur = cv2.fastNlMeansDenoising(frame, 7, 21)    
    blur = cv2.medianBlur(frame, 11)
    #blur = cv2.GaussianBlur(frame, (7, 7), 13)
    median_adaptive_th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 7)
    return median_adaptive_th

def median_th(frame):
    blur = cv2.medianBlur(frame, 11)
    median_th =  cv2.threshold(blur, 100, 255,
                                    cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    return median_th

def get_contours(frame):
    contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_contour(frame, contours):
    for cont in contours:
        cv2.drawContours(frame, cont, -1, (255, 0, 255), 2)
        (x,y,w,h) = cv2.boundingRect(cont)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (10,0,10), 2)
    return frame


try:
    cap = cv2.VideoCapture(0)    
    #cap.set(3, 1920)
    #cap.set(4, 1080)
except:
    print("Can't read video frame.")

while True:
#---------- BEGINING TO READ
    _, frame = cap.read()
    
    # -- -- -- BEGINING TO DO COMPUTING ON FRAMES
    begin = time.time()
    
    # -- -- -- create a copy from original frame  for illustration purposes
    original_frame = cv2.flip(frame.copy() ,1)
    
    # -- -- -- preprocess frames (fliping, bgr2gray, resize)
    frame = preprocess_frame(frame)
    
    # -- -- -- blur the frame and get a threshold
    threshold = median_th(frame)
    #threshold = median_adaptive_th(frame)
    #cv2.imshow("Threshold", threshold)
    
    # -- -- -- extract contours from the thresholded frame
    #cntrs = get_contours(median_threshold)
    cntrs = get_contours(threshold)
    
    # -- -- -- draw contours on the original frame
    original_frame = draw_contour(original_frame, cntrs)    
    
    # -- -- -- loop through contours and create a mask of True(s) and Flase(s)
    mask = np.ones(original_frame.shape[:2], dtype="uint8") * 255
    for cont in cntrs:
        mask = cv2.drawContours(mask, [cont], -1, 0, -1)
       
    cv2.imshow("mask", mask)
    
    
    
#---------- END OF FRAME COMPUTING PART
    end = time.time()
    fps = 1 /(end-begin)
    cv2.putText(original_frame, f"fps:{int(fps)}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (20,20,20), 2)
    cv2.imshow("feed", original_frame)    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break