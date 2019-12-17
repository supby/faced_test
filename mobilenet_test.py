# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 15:13:48 2019

@author: 11020083
"""

import cv2
import tensorflow as tf
import numpy as np

pb  = './mobilenet_v1_1.0_224_frozen.pb'
pbt = './mobilenet_v1_1.0_224_eval.pbtxt'

classes = ["background", "person", "bicycle", "car", "motorcycle",
    "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "unknown", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "unknown", "backpack",
    "umbrella", "unknown", "unknown", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "unknown", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "unknown", "dining table",
    "unknown", "unknown", "toilet", "unknown", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "unknown",
"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" ]

net = cv2.dnn.readNetFromTensorflow(pb, pbt)

cap = cv2.VideoCapture(0)

while(True):
    
    ret, frame = cap.read()
    
    rows = frame.shape[0]
    cols = frame.shape[1]
    
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0/127.5, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    outs = cvNet.forward()
    
    for detection in outs[0,0,:,:]:
        score = float(detection[2])
        if score > 0.3:
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)

            idx = int(detection[1])   # prediction class index. 
            
            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(classes[idx],score * 100)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(img, label, (int(left), int(y)),cv.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)
    
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()