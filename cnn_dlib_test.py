import sys
import dlib
import numpy as np
import cv2

cnn_face_detector = dlib.cnn_face_detection_model_v1('./mmod_human_face_detector.dat')

cap = cv2.VideoCapture(0)

while(True):
    
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces_cnn = cnn_face_detector(frame, 1)

    for face in faces_cnn:
        x = face.rect.left()
        y = face.rect.top()
        w = face.rect.right() - x
        h = face.rect.bottom() - y

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()