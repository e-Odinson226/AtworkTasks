from matplotlib import image, pyplot as plt
import time
import cv2

#---------- BEGINING TO READ
def preprocess_frame(addr):
    frame = cv2.imread(addr, 0)
    return cv2.resize(frame, (720, 480), interpolation= cv2.INTER_AREA)

#---------- BEGINING TO DO COMPUTING ON FRAMES
#frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
def compute_frame(frame):
    median_blur = cv2.medianBlur(frame, 11)
    median_denoised = cv2.fastNlMeansDenoising(median_blur, 10, 10, 7, 21)

    median_th =  cv2.threshold(median_blur, 100, 255,
                                    cv2.THRESH_OTSU)[1]
    median_adaptive_th = cv2.adaptiveThreshold(median_denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 21, 2)
    return [median_th, median_adaptive_th]

address_list = ['./img samples/01.jpg',
                './img samples/02.jpg',
                './img samples/03.jpg',
                './img samples/04.jpg',
                './img samples/05.jpg',]
images = []
for address in address_list:
    frame = preprocess_frame(address)
    computed_list = compute_frame(frame)
    images.append( {'title':'Frame',    'image': frame} )
    images.append( {'title':'Median Threshold', 'image': computed_list[0]} )
    images.append( {'title':'Median Adaptive Threshold', 'image': computed_list[1]} )
#---------- END OF FRAME COMPUTING PART
plt.figure(figsize=(16, 7))
for i in range(len(images)):
    plt.subplot(5,3,i+1),plt.imshow(images[i]['image'],'gray')
    plt.title(images[i]['title'])
    plt.xticks([]),plt.yticks([])
plt.show()