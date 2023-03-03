import numpy as np
import cv2


# frame = cv2.imread("../test_images/Axis_4.jpg")

cap = cv2.VideoCapture("color_Axis.avi")

fgbg = cv2.createBackgroundSubtractorKNN()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

while True:
    ret, frame = cap.read()
    frame_0 = frame.copy()
    frame_1 = frame.copy()

    foreground_mask = fgbg.apply(frame)
    foreground_mask_processed = cv2.morphologyEx(
        foreground_mask, cv2.MORPH_OPEN, kernel
    )

    # detect obj and draw contour

    contours_0, hierarchy_0 = cv2.findContours(
        foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    contours_1, hierarchy_1 = cv2.findContours(
        foreground_mask_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    try:
        hierarchy_0 = hierarchy_0[0]
    except:
        hierarchy_0 = []

    try:
        hierarchy_1 = hierarchy_1[0]
    except:
        hierarchy_1 = []

    height, width = foreground_mask.shape
    min_x, min_y = width, height
    max_x = max_y = 0

    # computes the bounding box for the contour, and draws it on the frame,
    for contour, hier in zip(contours_0, hierarchy_0):
        (x, y, w, h) = cv2.boundingRect(contour)
        min_x, max_x = min(x, min_x), max(x + w, max_x)
        min_y, max_y = min(y, min_y), max(y + h, max_y)
        if w > 80 and h > 80:
            cv2.rectangle(frame_0, (x, y), (x + w, y + h), (255, 0, 0), 2)

    if max_x - min_x > 0 and max_y - min_y > 0:
        cv2.rectangle(frame_0, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)

    for contour, hier in zip(contours_1, hierarchy_1):
        (x, y, w, h) = cv2.boundingRect(contour)
        min_x, max_x = min(x, min_x), max(x + w, max_x)
        min_y, max_y = min(y, min_y), max(y + h, max_y)
        if w > 80 and h > 80:
            cv2.rectangle(frame_1, (x, y), (x + w, y + h), (255, 0, 0), 2)

    if max_x - min_x > 0 and max_y - min_y > 0:
        cv2.rectangle(frame_1, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)

    cv2.imshow("frame_0", frame_0)
    cv2.imshow("frame_1", frame_1)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    if key == ord("p"):
        cv2.waitKey(-1)


cv2.destroyAllWindows()
