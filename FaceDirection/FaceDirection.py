import cv2
import mediapipe as mp
from numpy import result_type

# Create mediapipe modules and implement classes from it.
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh( 
                               max_num_faces=1,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5  )

# Read feed from webcam and resize
cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 720)

while True:
    isReadOk, frame = cap.read()
    if not(isReadOk):
        print("could'nt read the feed from webcam.")
        break
    
    # Edit input frame to be RGB
    # and flip it to be as a selfie.
    frame = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)
    # make it not writeable to procces faster.
    frame.flags.writeable = False
    
    # Process the frame.
    result = faceMesh.process(frame)
    
    # make it writeable.
    frame.flags.writeable = True
    
    # Convert it back to BGR to be able to work on it.
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Check if results have been aquierd then:
    # Loop trough landmarks.
    if result.multi_face_landmarks:
        for faceLandmark in result.multi_face_landmarks:
            print(faceLandmark)
    
    
    #cv2.imshow("Capture", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    cv2.imshow("Capture", frame)
    
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break