# First import library
import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import argparse
import os.path
import time


# Create object for parsing command-line options
parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                Remember to change the stream fps and format to match the recorded.")

# Add argument which takes path to a bag file as an input
parser.add_argument("-i", "--input", type=str, help="Path to the bag file")

# Parse the command line arguments to an object
args = parser.parse_args()

# Safety if no parameter have been given
if not args.input:
    print("No input paramater have been given.")
    print("For help type --help")
    exit()
# Check if the given file have bag extension
if os.path.splitext(args.input)[1] != ".bag":
    print("The given file is not of correct file format.")
    print("Only .bag files are accepted")
    exit()

kernel_RECT = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

def process(frame):

    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blured_frame = cv.GaussianBlur(frame, (13, 13), 10)
    # blur = cv.medianBlur(frame, 7)

    ret, threshold_frame = cv.threshold(
        blured_frame, 0, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C + cv.THRESH_OTSU
    )

    """ threshold_frame = cv.adaptiveThreshold(
        blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 61, 20
    ) """

    #kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    # threshold_frame = cv.dilate(threshold_frame, kernel, iterations=1)
    # out_frame = cv.erode(threshold_frame, kernel, iterations=1)

    out_frame = cv.morphologyEx(threshold_frame, cv.MORPH_OPEN, kernel_RECT)

    return out_frame

def auto_canny(frame, sigma=0.33):
    
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray_frame, (11, 11), 0)
    
    # compute the median of the single channel pixel intensities
    v = np.median(blurred)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv.Canny(blurred, lower, upper)
	
    # return the edged image
    return edged
 
 
def detect_contour(frame):
    contours, hierarchy = cv.findContours(
        frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        #frame, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE
    )
    # return (contours, hierarchy)
    return (contours, hierarchy)


def draw_objects(frame, contours, hierarchy):
    try:
        hierarchy = hierarchy[0]
    except:
        hierarchy = []

    height, width = frame.shape[:2]
    min_x, min_y = width, height
    max_x = max_y = 0

    # computes the bounding box for the contour, and draws it on the frame,
    for contour, hier in zip(contours, hierarchy):
        (x, y, w, h) = cv.boundingRect(contour)
        min_x, max_x = min(x, min_x), max(x + w, max_x)
        min_y, max_y = min(y, min_y), max(y + h, max_y)
        if w > 10 and h > 10:
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    if max_x - min_x > 0 and max_y - min_y > 0:
        cv.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 255, 0), 2)

    return frame

    """ cv.drawContours(frame, contours, -1, (0, 255, 0), 3)
    # cv.drawContours(frame, conts, -1, (22, 22, 22), 2, cv.LINE_8, hierarchy, 0)

    return frame """

kernel = np.ones((5, 5))

try:
    # Create pipeline
    pipeline = rs.pipeline()

    # Create a config object
    config = rs.config()

    # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
    rs.config.enable_device_from_file(config, args.input)

    # Configure the pipeline to stream the depth stream
    # Change this parameters according to the recorded bag file resolution
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, rs.format.rgb8, 30)
    

    # Start streaming from file
    pipeline.start(config)

    # Create opencv window to render image in
    cv.namedWindow("RGB Frame", cv.WINDOW_AUTOSIZE)
    
    # Create colorizer object
    colorizer = rs.colorizer()

    # Streaming loop
    while True:
        # Get frameset of depth
        frames = pipeline.wait_for_frames()

        # Get RGB frame
        begin = time.time()
        rgb_frame = frames.get_color_frame()
        rgb_frame = np.asanyarray(rgb_frame.get_data())
        # ////////////////////////////////////////////////////////////////////////////////
        
        processed_frame = process(rgb_frame)
        #contours, hierarchy = detect_contour(processed_frame)
        cv.imshow("processed_frame", processed_frame)
        
                
        #canny_frame = auto_canny(rgb_frame)
        #canny_contours, canny_hierarchy = detect_contour(canny_frame)
        #cv.imshow("canny_frame", canny_frame)
        # ////////////////////////////////////////////////////////////////////////////////
        
        #draw_objects(rgb_frame, contours, hierarchy)

        end = time.time()
        fps = 1 /(end-begin)
        cv.putText(rgb_frame, f"fps:{int(fps)}", (5, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (20,20,20), 2)

        #cv.imshow("RGB Frame", rgb_frame)
        
        
        key = cv.waitKey(1)
        if key == ord("q"):
            cv.destroyAllWindows()
            break
        if key == ord("p"):
            cv.waitKey(-1)

finally:
    pass
