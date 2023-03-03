import cv2

# frame = cv2.imread("../test_images/Axis_4.jpg")

cap = cv2.VideoCapture("color_Bearing.avi")
fgbg = cv2.createBackgroundSubtractorMOG2()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

while True:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)

    foreground_mask_processed = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    cv2.imshow("frame", foreground_mask_processed)

    if cv2.waitKey(30) & 0xFF == 27:
        break


cv2.destroyAllWindows()
