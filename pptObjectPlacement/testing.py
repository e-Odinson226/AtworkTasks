import cv2
import matplotlib.pyplot as plt

frame = cv2.imread('./img samples/01.jpg', 0)
median_blur = cv2.medianBlur(frame, 11)
median_th =  cv2.threshold(median_blur, 100, 255,
                                    cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
_, cntrs = cv2.findContours(median_th)
while True:
    _, frame = cap.read()
    plt.figure(figsize=(2, 7))
    plt.subplot(1,2,1),plt.imshow(frame, 'gray')
    plt.title('title')
    plt.xticks([]),plt.yticks([])
    plt.show()