import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
#cap2 = cv2.VideoCapture(1)

#_, frame2 = cap2.read()
#frames = [frame, frame2]
#cv2.imshow('cap', frame)
while True:
    _, frame = cap.read()
    plt.figure(figsize=(2, 7))
    plt.subplot(1,2,1),plt.imshow(frame, 'gray')
    plt.title('title')
    plt.xticks([]),plt.yticks([])
    plt.show()