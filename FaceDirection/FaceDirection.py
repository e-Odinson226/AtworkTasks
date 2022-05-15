from traceback import print_tb
import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
cap.set(3, 720)
cap.set(4, 1080)

while True:
    isReadOk, frame = cap.read()
    if not(isReadOk):
        print("could'nt read the feed from webcam.")
        break
    
    frame = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)
    
    frame.flags.writeable = False
    cv2.imshow("Capture", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break