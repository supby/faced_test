import cv2
import tensorflow as tf
import numpy as np

modelFile = "./opencv_face_detector_uint8.pb"
configFile = "./opencv_face_detector.pbtxt"

net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()

    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.3:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (23, 230, 210), thickness=2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
