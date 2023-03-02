import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tools import show_image_list




def __init__(self):
    self.backGroundSubMOG2 = cv2.createBackgroundSubtractorMOG2()
    self.backGroundSubKNN = cv2.createBackgroundSubtractorKNN()

def doG(self, frame, kernel_0, sigma_0, kernel_1, sigma_1):
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
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)

    # frame = cv2.imread("../test_images/Axis_5.jpg")

    # ----------------------------------------
    ax0 = plt.subplot(2, 2, 1)
    plt.title("frame")
    ax1 = plt.subplot(2, 2, 2)
    plt.title("DOG")
    ax2 = plt.subplot(2, 2, 3)
    plt.title("KNN")
    ax3 = plt.subplot(2, 2, 4)
    plt.title("MOG2")
    # ----------------------------------------

    bgd = BackgroundDetector()

    # backgroundframeDOG = bgd.doG(frame, 7, 7, 17, 13)
    # backgroundframeKNN = bgd.kNN(frame)
    # backgroundframeMOG2 = bgd.moG2(frame)

    # ----------------------------------------
    im0 = ax0.imshow(get_frame(cap))
    im1 = ax1.imshow(bgd.doG(get_frame(cap), 7, 7, 17, 13))
    im2 = ax2.imshow(bgd.kNN(get_frame(cap)))
    im3 = ax3.imshow(bgd.moG2(get_frame(cap)))
    # ----------------------------------------

    def update(i):

        im0.set_data(get_frame(cap))
        im1.set_data(bgd.doG(get_frame(cap), 7, 7, 17, 13))
        im2.set_data(bgd.kNN(get_frame(cap)))
        im3.set_data(bgd.moG2(get_frame(cap)))

    ani = FuncAnimation(plt.gcf(), update, interval=200)
    plt.show()
