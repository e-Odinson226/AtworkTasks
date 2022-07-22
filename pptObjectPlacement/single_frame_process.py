from cv2 import waitKey
from matplotlib import pyplot as plt
import time
import cv2
import numpy as np

def preprocess_frame(frame):
    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return frame

def median_adaptive_th(frame):
    #blur = cv2.fastNlMeansDenoising(frame, 7, 21)    
    blur = cv2.medianBlur(frame, 13)
    #blur = cv2.GaussianBlur(frame, (7, 7), 13)
    median_adaptive_th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 17, 7)
    return median_adaptive_th

def median_th(frame):
    blur = cv2.medianBlur(frame, 11)
    median_th =  cv2.threshold(blur, 100, 255,
                                    cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    return median_th

def get_contours(frame):
    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_rectangle(frame, contours):
    error_persent = 1.30
    for cont in contours:
        #cv2.drawContours(frame, cont, -1, (255, 0, 255), 2)
        (x,y,w,h) = cv2.boundingRect(cont)
        w = int(w * error_persent)
        h = int(h * error_persent)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (10,0,10), 2)
    return frame

def create_mask(mask, contours):
    # Create a +1.3 coefficient as predicted error
    error_persent = 1.30
    for cont in contours:
        (x,y,w,h) = cv2.boundingRect(cont)
        w = int(w * error_persent)
        h = int(h * error_persent)
        cv2.rectangle(mask, (x, y), (x+w, y+h), 0, -1)
    return mask


address_list = ['./img samples/01.jpg',
                './img samples/02.jpg',
                './img samples/03.jpg',
                './img samples/04.jpg',
                './img samples/05.jpg',
                ]

while True:
#---------- BEGINING TO READ
    frame = cv2.imread(address_list[2])
    frame = cv2.resize(frame, (720, 640), interpolation= cv2.INTER_AREA)
# -- -- -- BEGINING TO DO COMPUTING ON FRAMES
    begin = time.time()

    # -- -- -- create a copy from original frame  for illustration purposes
    original_frame = cv2.flip(frame.copy() ,1)

    # -- -- -- preprocess frames (fliping, bgr2gray, resize)
    frame = preprocess_frame(frame)

    # -- -- -- blur the frame and get a threshold
    threshold = median_th(frame)
    #threshold = median_adaptive_th(frame)
    cv2.imshow("Threshold", threshold)

    # -- -- -- extract contours from the thresholded frame
    #cntrs = get_contours(median_threshold)
    cntrs = get_contours(threshold)
    
    tmpFrame = original_frame.copy()
    # -- -- -- draw contours on the original frame
    original_frame = draw_rectangle(original_frame, cntrs)
    
    # -- -- -- loop through contours and create a mask of True(s) and Flase(s)
    mask = np.ones(frame.shape[:2], dtype="uint8") * 255
    
    mask = create_mask(mask, cntrs)
    masked_frame = cv2.bitwise_and(tmpFrame, tmpFrame, mask=mask)

    cv2.imshow("mask", mask)
    cv2.imshow("Original", original_frame)
    cv2.imshow("masked_frame", masked_frame)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break