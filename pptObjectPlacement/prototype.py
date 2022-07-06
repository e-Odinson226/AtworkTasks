from matplotlib import image, pyplot as plt
import time
import cv2

#---------- BEGINING TO READ
def preprocess_frame(addr):
    frame = cv2.imread(addr, 0)
    return cv2.resize(frame, (720, 480), interpolation= cv2.INTER_AREA)

#---------- BEGINING TO DO COMPUTING ON FRAMES
begin = time.time()

#---------- CHOOSE THRESHOLD METHOD
#frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
def compute_frame(frame):
    frameMedian = cv2.medianBlur(frame, 11)
    frameMedDenoised = cv2.fastNlMeansDenoising(frameMedian, 10, 10, 7, 21)

    frameMedTH =  cv2.threshold(frameMedian, 100, 255,
                                    cv2.THRESH_OTSU)[1]
    frameMedDenoisedTH = cv2.adaptiveThreshold(frameMedDenoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 21, 2)
    return [frameMedTH, frameMedDenoisedTH]

frame1 = compute_frame(preprocess_frame('./img samples/IMG_20220705_165233.jpg'))
#---------- END OF FRAME COMPUTING PART
images = [  {'i': frame1[0], 't':'frameMedTH'},
            {'i': frame1[1], 't':'frameMedDenoisedTH'}]
plt.figure(figsize=(16, 7))
for i in range(len(images)):
    plt.subplot(4,3,i+1),plt.imshow(images[i]['i'],'gray')
    plt.title(images[i]['t'])
    plt.xticks([]),plt.yticks([])
plt.show()