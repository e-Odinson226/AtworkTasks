import cv2 as cv


def dog(frame, low_kernel, low_sigma, high_kernel, high_sigma):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    low_gaussianBlur = cv.GaussianBlur(frame, (low_kernel, low_kernel), low_sigma)
    high_gaussianBlur = cv.GaussianBlur(frame, (high_kernel, high_kernel), high_sigma)
    return low_gaussianBlur - high_gaussianBlur


def draw_rect_for_contours(frame, contours, hierarchy):
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
        if w > 80 and h > 80:
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    if max_x - min_x > 0 and max_y - min_y > 0:
        cv.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 255, 0), 2)

    return frame


def nothing(x):
    pass


def trackbar(blur_modes={"bilateral": 0, "gaussian": 1}, dog_params=(3, 0, 5, 0)):
    low_kernel, low_sigma, high_kernel, high_sigma = dog_params
    cv.namedWindow("Detection Tweaks")

    cv.createTrackbar(
        "Blur Modes",
        "Detection Tweaks",
        blur_modes["bilateral"],
        blur_modes["gaussian"],
        nothing,
    )

    cv.createTrackbar("low_kernel", "Detection Tweaks", low_kernel, 10, nothing)
    cv.createTrackbar("low_sigma", "Detection Tweaks", low_sigma, 50, nothing)
    cv.createTrackbar("high_kernel", "Detection Tweaks", high_kernel, 10, nothing)
    cv.createTrackbar("high_sigma", "Detection Tweaks", high_sigma, 70, nothing)


if __name__ == "__main__":
    trackbar()
    address = (
        "/home/zakaria/Documents/Projects/AtworkTasks/dataset/video/color/video.avi"
    )
    kernel_CROSS = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    kernel_RECT = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    params = [(11, 21, 7), (11, 41, 21), (11, 61, 39)]

    # read frame
    cap = cv.VideoCapture(address)
    success, frame = cap.read()

    while success:
        low_kernel = cv.getTrackbarPos("low_kernel", "Detection Tweaks")
        low_kernel = low_kernel if low_kernel % 2 == 1 else low_kernel - 1

        low_sigma = cv.getTrackbarPos("low_sigma", "Detection Tweaks")

        high_kernel = cv.getTrackbarPos("high_kernel", "Detection Tweaks")
        high_kernel = high_kernel if high_kernel % 2 == 1 else high_kernel - 1

        high_sigma = cv.getTrackbarPos("high_sigma", "Detection Tweaks")

        foreground_mask = dog(frame, low_kernel, low_sigma, high_kernel, high_sigma)
        cv.imshow("foreground_mask", foreground_mask)

        blur_mode = cv.getTrackbarPos("Blur Modes", "Detection Tweaks")
        if blur_mode == 0:
            diameter, sigmaColor, sigmaSpace = params[1]
            blured_frame = cv.bilateralFilter(
                foreground_mask, diameter, sigmaColor, sigmaSpace
            )
        elif blur_mode == 1:
            blured_frame = cv.GaussianBlur(foreground_mask, (9, 9), 3)
        cv.imshow("blured_frame", blured_frame)

        # Process frame
        processed_foreground_mask = cv.morphologyEx(
            blured_frame, cv.MORPH_OPEN, kernel_RECT, iterations=1
        )
        cv.imshow("processed_foreground_mask", processed_foreground_mask)

        # contours, hierarchy = cv.findContours(
        #    foreground_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        # )

        # draw_rect_for_contours(
        #    frame,
        #    contours,
        #    hierarchy,
        # )

        success, frame = cap.read()
        key = cv.waitKey()
        if key == ord("q"):
            break
        if key == ord("p"):
            cv.waitKey(-1)


cv.destroyAllWindows()
