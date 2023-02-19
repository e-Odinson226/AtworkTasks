import cv2


def doG(frame, kernel_0, sigma_0, kernel_1, sigma_1):
    gaussianBlur_0 = cv2.GaussianBlur(frame, (kernel_0, kernel_0), sigma_0)
    gaussianBlur_1 = cv2.GaussianBlur(frame, (kernel_1, kernel_1), sigma_1)
    return gaussianBlur_0 - gaussianBlur_1


if __name__ == "__main__":

    # frame = cv2.imread("../test_images/Axis_5.jpg")
    cap = cv2.VideoCapture("video.avi")

    while True:
        ret, frame = cap.read()
        backgroundframeDOG = doG(frame, 7, 7, 17, 13)

        cv2.imshow("Background Detection with DOG", backgroundframeDOG)

        if cv2.waitKey(30) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()
