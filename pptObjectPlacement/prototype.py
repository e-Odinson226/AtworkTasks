import tk
from matplotlib import image, pyplot as plt

import time
import cv2

#---------- BEGINING TO READ
frame = cv2.imread('./img samples/IMG_20220705_165233.jpg')
frame = cv2.resize(frame, (720, 480), interpolation= cv2.INTER_AREA)
cv2.imshow('frame', frame)

#---------- BEGINING TO DO COMPUTING ON FRAMES
begin = time.time()

#---------- CHOOSE THRESHOLD METHOD
images = []

frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

frameBlured = cv2.GaussianBlur(frameGray, (11, 11), 0)
images.append(frameBlured)
#cv2.imshow("frameBL", frameBlured)

frameDenoised = cv2.fastNlMeansDenoising(frameGray, 10, 10, 7, 21)
images.append(frameDenoised)
#cv2.imshow("frameDN", frameDenoised)

#---------- END OF FRAME COMPUTING PART
fig = plt.figure(figsize=(14, 5))

for i, img in zip([ x for x in range(0, len(images))], images):
    fig.add_subplot(1, 4, i+1)
    plt.imshow(img)
    plt.axis('off')
    plt.title("img")
plt.show()
#fig.add_subplot(1, 4, 1)
#plt.imshow(frameBlured)
#plt.title("frameBlured")
#
#fig.add_subplot(1, 4, 1)
#plt.imshow(frameBlured)
#plt.title("frameBlured")
#
#fig.add_subplot(1, 4, 1)
#plt.imshow(frameBlured)
#plt.title("frameBlured")