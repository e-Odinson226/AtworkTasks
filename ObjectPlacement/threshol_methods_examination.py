from matplotlib import pyplot as plt
import cv2

#---------- BEGINING TO READ
def preprocess_frame(frame):
    #frame = cv2.imread(addr, 0)
    return cv2.resize(frame, (720, 480), interpolation= cv2.INTER_AREA)

def median_adaptive_th(frame):
    #median_denoised = cv2.fastNlMeansDenoising(frame, 7, 21)    
    median_blur = cv2.medianBlur(frame, 7)
    #gaus_blur = cv2.GaussianBlur(frame, (9, 9), 0)
    median_adaptive_th = cv2.adaptiveThreshold(median_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 5)
    return median_adaptive_th

def median_th(frame):
    median_blur = cv2.medianBlur(frame, 11)
    median_th =  cv2.threshold(median_blur, 100, 255,
                                    cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    return median_th

def get_contours(frame, drawOnFrame):
    contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cont in contours:
        cv2.drawContours(drawOnFrame, cont, -1, (255, 0, 255), 2)
        (x,y,w,h) = cv2.boundingRect(cont)
        cv2.rectangle(drawOnFrame, (x, y), (x+w, y+h), (10,0,10), 2)
            
    return drawOnFrame
            

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
    frame_median_adaptive_th = median_adaptive_th(frame)
    frame_median_th = median_th(frame)
    contoured_adaptiveTh = get_contours(frame_median_adaptive_th, frame.copy())
    contoured_medianTh = get_contours(frame_median_th, frame.copy())
    frames.append( {'title':'Frame',    'image': frame} )
    frames.append( {'title':'Median Threshold', 'image': frame_median_th} )
    frames.append( {'title':'Median Adaptive Threshold', 'image': frame_median_adaptive_th} )
    frames.append( {'title':'Median Threshold', 'image': contoured_medianTh} )
    frames.append( {'title':'Median Adaptive Threshold', 'image': contoured_adaptiveTh} )
    #frames.append(img)

#---------- END OF FRAME COMPUTING PART
plt.figure(figsize=(25, 7))
for i in range(len(frames)):
    plt.subplot(5,5,i+1),plt.imshow(frames[i]['image'],'gray')
    plt.title(frames[i]['title'])
    plt.xticks([]),plt.yticks([])
plt.show()