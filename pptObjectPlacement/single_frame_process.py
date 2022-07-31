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
    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

def draw_rectangle(frame, contours):
    error_persent = 1.30
    for cont in contours:
        #cv2.drawContours(frame, cont, -1, (255, 0, 255), 2)
        (x,y,w,h) = cv2.boundingRect(cont)
        w = int(w * error_persent)
        h = int(h * error_persent)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,255), 2)
    return frame

def create_mask(mask, contours):
    (_, _, w, h) = cv2.boundingRect(contours[0])
    min_contour = [w, h]
    min_contour_area = cv2.contourArea(contours[0])
    # Set a +1.3 coefficient as predicted error
    error_persent = 1.30
    
    for cont in contours:
        contour_area = cv2.contourArea(cont)
        #print(f"contour_area:{contour_area} threshold:{(mask.shape[0]*mask.shape[1])*0.002}")
        
        #   Filtering based on size
        #if contour_area > (mask.shape[0]*mask.shape[1])*0.002:
        
        #   GET MINIMUM AREA RECANGLE
        #rect =  cv2.minAreaRect(cont)
        #box = cv2.boxPoints(rect)
        #box = np.int0(box)
        #cv2.drawContours(mask, [box], 0, -1, -1)
        
        (x,y,w,h) = cv2.boundingRect(cont)
        w = int(w * error_persent)
        h = int(h * error_persent)
        cv2.rectangle(mask, (x, y), (x+w, y+h), 0, -1)
        if contour_area < min_contour_area:
            min_contour = [w, h]
            # TESTING PORPUSES
            #minCont = cont
    # TESTING PORPUSES
    #(x,y,w,h) = cv2.boundingRect(minCont)
    #w = int(w * error_persent)
    #h = int(h * error_persent)
    #cv2.rectangle(mask, (x, y), (x+w, y+h), 0, -1)

    return [mask, min_contour]

def convert_cm2pixle(dimention, lens):
    width = dimention['width']
    height = dimention['height']
    p_height =  height  * lens['focal_length'] / (lens['pixel_pitch'] * lens['lens_distance'])
    p_width =   width   * lens['focal_length'] / (lens['pixel_pitch'] * lens['lens_distance'])
    #pixels = real_distance * lens['focal_length'] / (lens['pixel_pitch'] * lens['lens_distance'])
    return {'width':p_width, 'height': p_height}

def create_template(dimention):
    return np.ones((dimention['height'], dimention['width']), dtype="uint8") * 255

def grid(frame, cell_width, cell_height):
    frame_h, frame_w = frame.shape[:2]
    grid_w = int(frame_w / cell_width)
    grid_h = int(frame_h / cell_height)
    grid = np.ones((grid_h, grid_w), dtype="uint8") * 255
    
    print(frame.shape[:2])
    print(cell_height, cell_width)
    print(grid_w)
    print(grid_h)
    cv2.imshow('grid', grid)
    for h in range(0, frame_h, cell_height):
        for w in range(0, frame_w, cell_width, ):            
            if (frame[h:h+cell_height, w:w+cell_width].any())==0:
                grid[int((h/cell_height)-1), int((w/cell_width)-1)] = 0
            #    print(f"----{int((h/cell_height)-1), int((w/cell_width)-1)}")
            else:
                grid[int((h/cell_height)-1), int((w/cell_width)-1)] = 255
            #    print(f"----{int((h/cell_height)-1), int((w/cell_width)-1)}")
                
    return grid
    

#def arrange(fov_frame, obj_frame, origin):
#    fovFrame_w, fovFrame_h = fov_frame.shape[::-1]
#    objFrame_w, objFrame_h = obj_frame.shape[::-1]
#    # check sides of object frame for admissible points on fov frame
#    ## des diameter:
#    ## asc diameter:
#    ## top:
#    ## right:
#    ## left:
#    ## bottom:
#
#    # if there was no interfere then check all pixels:
#    for y in range(origin[0], 10):
#        for x in range(origin[1], 10):
#            #print(f"{x}*{y}: {fov_frame[y, x]}")
#            if fov_frame[y, x] == 255:
#            #    print(True)
#            #else:   i+=1

#def scale_ratio(frame, lens):
#    real_distance = lens['pixel_pitch'] * pixels * lens['lens_distance'] / lens['focal_length']
    
# Implementation with example frames -----------------------------------------
address_list = ['./img samples/01.jpg',
                './img samples/02.jpg',
                './img samples/03.jpg',
                './img samples/04.jpg',
                './img samples/05.jpg',
                ]
#lens = {
#    'focal_length'  : 4.7 mili,
#    'pixel_pitch'   : 2.2 micro,
#    'lens_distance' : 0,
#}

obj_dim = [
    { 'width':150    ,'height':50 },
    { 'width':5    ,'height':10 },
    { 'width':8    ,'height':8 }
]
# temporary set pixle dimentions as static
p_dim = {'width':80, 'height': 400}

while True:
#---------- BEGINING TO READ
    frame = cv2.imread(address_list[1])
    frame = cv2.resize(frame, (1280, 720), interpolation= cv2.INTER_AREA)
# -- -- -- BEGINING TO DO COMPUTING ON FRAMES
    begin = time.time()

    # -- -- -- create a copy from original frame  for illustration purposes
    original_frame = cv2.flip(frame.copy() ,1)

    # -- -- -- preprocess frames (fliping, bgr2gray, resize)
    frame = preprocess_frame(frame)

    # -- -- -- blur the frame and get a threshold
    threshold = median_th(frame)
    #threshold = median_adaptive_th(frame)

    # -- -- -- extract contours from the thresholded frame
    cntrs = get_contours(threshold)
    
    # -- -- -- draw contours on the original frame
    original_frame = draw_rectangle(original_frame, cntrs)
    
    # -- -- -- loop through contours and create a mask of True(s) and Flase(s)
    #print(frame.shape[:2])
    mask = np.ones(frame.shape[:2], dtype="uint8") * 255
    mask, min_contour = create_mask(mask, cntrs)
    grid_cell_w = min_contour[0]
    grid_cell_h = min_contour[1]
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    
    # -- -- -- convert given size in cm to pixles
    #p_dim = convert_cm2pixle(obj_dim[0], lens)
    #draw_rectangle(original_frame, [p_dim])
    
    # -- -- -- template matching
    #template = create_template(p_dim)
    #w, h = template.shape[::-1]
    #res = cv2.matchTemplate(mask, template, cv2.TM_CCOEFF)
    #templateMatching_threshold = 0.8
    #loc = np.where( res >= templateMatching_threshold)
    #for pt in zip(*loc[::-1]):
    #    cv2.rectangle(masked_frame, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    
    # -- -- -- CREATE GRID FRAME AND ASSIGN VALUES FOR EACH GRID CELL
    #print(f"object:{obj_dim[0]['width'], obj_dim[0]['height']}")
    #grid_frame = grid(mask, obj_dim[0]['width'], obj_dim[0]['height'])
    grid_frame = grid(mask, grid_cell_w, grid_cell_h)
       
    
    
    print("----------------------------------------------------------")
    print("{img} shape: {shape}, dataType:{dtype}".format(img='mask', shape=mask.shape, dtype=mask.dtype))
    print("{img} shape: {shape}, dataType:{dtype}".format(img='original_frame', shape=original_frame.shape, dtype=original_frame.dtype))
    print("{img} shape: {shape}, dataType:{dtype}".format(img='masked_frame', shape=masked_frame.shape, dtype=masked_frame.dtype))
    cv2.imshow("mask", mask)
    cv2.imshow("Original", original_frame)
    cv2.imshow("masked_frame", masked_frame)
    cv2.imshow("grid_frame", grid_frame)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break