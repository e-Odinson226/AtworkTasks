import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tools import show_image_list

# def gsoc(self):
#    cv.bgse


def doG(frame, kernel_0, sigma_0, kernel_1, sigma_1):
    gaussianBlur_0 = cv2.GaussianBlur(frame, (kernel_0, kernel_0), sigma_0)
    gaussianBlur_1 = cv2.GaussianBlur(frame, (kernel_1, kernel_1), sigma_1)
    return gaussianBlur_0 - gaussianBlur_1


def moG2(self, frame):
    img = self.backGroundSubMOG2.apply(frame)
    return img


def kNN(self, frame):
    img = self.backGroundSubKNN.apply(frame)
    return img


def gMG(self, frame):
    pass


def get_frame(cap):
    ret, frame = cap.read()
    # return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


if __name__ == "__main__":

    cap = cv2.VideoCapture("video.avi")

    bgsMOG2 = cv2.createBackgroundSubtractorMOG2()
    bgsKNN = cv2.createBackgroundSubtractorKNN()

    # ----------------------------------------
    ax0 = plt.subplot(2, 2, 1)
    plt.title("frame")
    ax1 = plt.subplot(2, 2, 2)
    plt.title("DOG")
    ax2 = plt.subplot(2, 2, 3)
    plt.title("MOG2")
    ax3 = plt.subplot(2, 2, 4)
    plt.title("KNN")

    # ----------------------------------------
    im0 = ax0.imshow(get_frame(cap))
    im1 = ax1.imshow(doG(get_frame(cap), 7, 7, 17, 13))
    im2 = ax2.imshow(bgsMOG2.apply(get_frame(cap)))
    im3 = ax3.imshow(bgsKNN.apply(get_frame(cap)))

    def update(i):
        im0.set_data(get_frame(cap))
        im1.set_data(doG(get_frame(cap), 7, 7, 17, 13))
        im2.set_data(bgsMOG2.apply(get_frame(cap)))
        im3.set_data(bgsKNN.apply(get_frame(cap)))

    ani = FuncAnimation(plt.gcf(), update, interval=1)
    plt.show()
