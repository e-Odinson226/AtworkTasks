from matplotlib import pyplot as plt
import time
import cv2 as cv
import numpy as np


def process(frame):
    blur = cv.GaussianBlur(frame, (11, 11), 0)
    blur = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
    # blur = cv.medianBlur(frame, 7)

    ret, threshold_frame = cv.threshold(
        blur, 0, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C + cv.THRESH_OTSU
    )

    """ threshold_frame = cv.adaptiveThreshold(
        blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 61, 20
    ) """

    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    # threshold_frame = cv.dilate(threshold_frame, kernel, iterations=1)
    # out_frame = cv.erode(threshold_frame, kernel, iterations=1)

    out_frame = cv.morphologyEx(threshold_frame, cv.MORPH_GRADIENT, kernel)

    return out_frame


def detect_contour(frame):
    contours, hierarchy = cv.findContours(
        frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    # return (contours, hierarchy)
    return contours


def draw_objects(frame, contours):
    error_persent = 1.30
    (xm, ym, wm, hm) = cv.boundingRect(contours[0])
    wm = int(wm * error_persent)
    hm = int(hm * error_persent)

    min_contour_area = cv.contourArea(contours[0])
    for cont in contours:
        contour_area = cv.contourArea(cont)
        (x, y, w, h) = cv.boundingRect(cont)
        w = int(w * error_persent)
        h = int(h * error_persent)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
        if contour_area < min_contour_area:
            (xm, ym, wm, hm) = (x, y, w, h)

    return frame

    """ cv.drawContours(frame, contours, -1, (0, 255, 0), 3)
    # cv.drawContours(frame, conts, -1, (22, 22, 22), 2, cv.LINE_8, hierarchy, 0)

    return frame """


if __name__ == "__main__":
    address = (
        "/home/zakaria/Documents/Projects/AtworkTasks/dataset/video/color/video.avi"
    )
    # read frame
    cap = cv.VideoCapture(address)
    success, frame = cap.read()

    while success:
        # Process frame
        processed_frame = process(frame)

        # Display Processed frame
        cv.imshow("processed_frame", processed_frame)

        # detect contours
        contours = detect_contour(processed_frame)

        # detect objects
        # objects = detect(contours, frame)

        # draw rectangle around objects
        mask = draw_objects(frame, contours)
        cv.imshow("output_frame", mask)

        success, frame = cap.read()
        key = cv.waitKey()
        if key == ord("q"):
            break
        if key == ord("p"):
            cv.waitKey(-1)


cv.destroyAllWindows()
