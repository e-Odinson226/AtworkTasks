import cv2 
frame = cv2.imread('./img samples/02.jpg')
frame_copy = frame.copy()
contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cont in contours:
    cv2.drawContours(frame_copy, cont, -1, (255, 0, 255), 2)
    (x,y,w,h) = cv2.boundingRect(cont)
    cv2.rectangle(frame_copy, (x, y), (x+w, y+h), (10,0,10), 2)
cv2.imshow('test', frame_copy)