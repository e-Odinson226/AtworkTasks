# First import library
import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import argparse
import os.path
import time
import imutils


# Create object for parsing command-line options
parser = argparse.ArgumentParser(
    description="Read recorded bag file and display depth stream in jet colormap.\
                                Remember to change the stream fps and format to match the recorded."
)

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

kernel_MORPH_RECT = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
kernel_MORPH_CROSS = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
kernel_MORPH_ELLIPSE = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))


def process(
    frame,
    erode_iter=3,
    dilate_iter=4,
    morph_kernel=kernel_MORPH_CROSS,
    gaussian_kernel_size=13,
):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blured_frame = cv.GaussianBlur(
        frame, (gaussian_kernel_size, gaussian_kernel_size), 0
    )
    # blur = cv.medianBlur(frame, 7)

    ret, threshold_frame = cv.threshold(
        blured_frame, 0, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C + cv.THRESH_OTSU
    )

    """ threshold_frame = cv.adaptiveThreshold(
        blured_frame, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 61, 20
    ) """

    threshold_frame = cv.erode(threshold_frame, morph_kernel, iterations=erode_iter)
    out_frame = cv.dilate(threshold_frame, morph_kernel, iterations=dilate_iter)

    # out_frame = cv.morphologyEx(threshold_frame, cv.MORPH_OPEN, kernel_RECT)

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
        frame,
        # cv.RETR_EXTERNAL,
        # cv.RETR_LIST,
        cv.RETR_TREE,
        cv.CHAIN_APPROX_SIMPLE,
    )
    # cv.drawContours(rgb_frame, contours, -1, (255, 22, 22), 4)

    return contours


def draw_objects(frame, contours):
    # computes the bounding box for the contour, and draws it on the frame,
    for contour in contours:
        peri = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, 0.04 * peri, True)

        if len(approx) < 8 and cv.contourArea(contour) > 250:
            M = cv.moments(contour)
            x_center = int((M["m10"] / M["m00"]))
            y_center = int((M["m01"] / M["m00"]))

            # cv.drawContours(frame, [contour], 0, (0, 0, 0), 6)
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 4)

        # {"x_center": x_center, "y_center": y_center}
    return frame


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
    # cv.namedWindow("RGB Frame", cv.WINDOW_AUTOSIZE)

    # Create colorizer object
    colorizer = rs.colorizer()

    # Create plot to show result in image form

    # Streaming loop
    while True:
        frames = pipeline.wait_for_frames()

        # /////////////////////////////////  Get RGB frame /////////////////////////////////
        begin = time.time()
        rgb_frame = frames.get_color_frame()
        rgb_frame = np.asanyarray(rgb_frame.get_data())

        # /////////////////////////////////  Processing frames /////////////////////////////////
        processed_frame = process(rgb_frame)
        # canny_frame = auto_canny(rgb_frame)

        # /////////////////////////////////  Find contours and Draw /////////////////////////////////
        # canny_contours, canny_hierarchy = detect_contour(canny_frame)
        contours = detect_contour(processed_frame)
        rgb_frame = draw_objects(rgb_frame, contours)

        end = time.time()
        fps = 1 / (end - begin)

        cv.putText(
            rgb_frame,
            f"fps:{int(fps)}",
            (5, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 250, 0),
            2,
        )

        cv.imshow("RGB Frame", rgb_frame)

        key = cv.waitKey(1)
        if key == ord("q"):
            cv.destroyAllWindows()
            break
        if key == ord("p"):
            cv.waitKey(-1)

finally:
    pass
