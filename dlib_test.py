import sys
import dlib
import numpy as np
import cv2

detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture(0)

while(True):
    
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    dets = detector(gray, 1)

    for i, d in enumerate(dets):
        cv2.rectangle(frame, (d.left(),d.top()), (d.right(),d.bottom()), (0,255,0), 2)
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()