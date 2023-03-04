import numpy as np
import cv2 as cv


# frame = cv.imread("../test_images/Axis_4.jpg")

cap = cv.VideoCapture(
    "/home/zakaria/Documents/Projects/AtworkTasks/dataset/video/color/video.avi"
)

fgbg = cv.createBackgroundSubtractorKNN()
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

while True:
    ret, frame = cap.read()

    foreground_mask = fgbg.apply(frame)
    foreground_mask_processed = cv.morphologyEx(
        foreground_mask, cv.MORPH_GRADIENT, kernel
    )

    # detect obj and draw contour

    contours_0, hierarchy_0 = cv.findContours(
        foreground_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )

    try:
        hierarchy_0 = hierarchy_0[0]
    except:
        hierarchy_0 = []

    height, width = foreground_mask.shape
    min_x, min_y = width, height
    max_x = max_y = 0

    # computes the bounding box for the contour, and draws it on the frame,
    for contour, hier in zip(contours_0, hierarchy_0):
        (x, y, w, h) = cv.boundingRect(contour)
        min_x, max_x = min(x, min_x), max(x + w, max_x)
        min_y, max_y = min(y, min_y), max(y + h, max_y)
        if w > 80 and h > 80:
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    if max_x - min_x > 0 and max_y - min_y > 0:
        cv.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)

    cv.imshow("frame", frame)

    key = cv.waitKey()
    if key == ord("q"):
        break
    if key == ord("p"):
        cv.waitKey(-1)


cv.destroyAllWindows()
