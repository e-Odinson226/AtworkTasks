from matplotlib import image, pyplot as plt
import time
import cv2

#---------- BEGINING TO READ
def preprocess_frame(frame):
    #frame = cv2.imread(addr, 0)
    return cv2.resize(frame, (720, 480), interpolation= cv2.INTER_AREA)

def compute_frame(frame):
    median_blur = cv2.medianBlur(frame, 11)
    median_denoised = cv2.fastNlMeansDenoising(median_blur, 10, 10, 7, 21)    

    median_th =  cv2.threshold(median_blur, 100, 255,
                                    cv2.THRESH_OTSU)[1]
    median_adaptive_th = cv2.adaptiveThreshold(median_denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 21, 2)
    return [median_th, median_adaptive_th]

def get_contours(frame):
    frame_copy = frame.copy()
    contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cont in contours:
        cv2.drawContours(frame_copy, cont, -1, (255, 0, 255), 2)
        (x,y,w,h) = cv2.boundingRect(cont)
        cv2.rectangle(frame_copy, (x, y), (x+w, y+h), (10,0,10), 2)
            
    return frame_copy
            

#---------- BEGINING TO DO COMPUTING ON FRAMES
address_list = ['./img samples/01.jpg',
                './img samples/02.jpg',
                './img samples/03.jpg',
                './img samples/04.jpg',
                './img samples/05.jpg',
                ]
frames = []
for address in address_list:
    #img = []
    frame = cv2.imread(address, 0)
    frame = preprocess_frame(frame)
    computed_list = compute_frame(frame)
    contoured_imagelist = get_contours(computed_list[0])
    frames.append( {'title':'Frame',    'image': frame} )
    frames.append( {'title':'Median Threshold', 'image': computed_list[0]} )
    frames.append( {'title':'Median Adaptive Threshold', 'image': computed_list[1]} )
    frames.append( {'title':'contoured', 'image': contoured_imagelist} )
    #frames.append(img)

#---------- END OF FRAME COMPUTING PART
plt.figure(figsize=(20, 7))
for i in range(len(frames)):
    plt.subplot(5,4,i+1),plt.imshow(frames[i]['image'],'gray')
    plt.title(frames[i]['title'])
    plt.xticks([]),plt.yticks([])
plt.show()