# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:41:48 2019

@author: 11020083
"""

import cv2
import numpy as np

config = 'yolov3-tiny.cfg'
model = 'yolov3-tiny.weights'

net = cv2.dnn.readNetFromDarknet(config, model)

cap = cv2.VideoCapture(0)

while(True):
    
    ret, frame = cap.read()
    
    rows = frame.shape[0]
    cols = frame.shape[1]
    
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0/255.0, (416, 416), swapRB = True, crop = False))

    outs = net.forward()
    
    print(">>> YOLOv3 tiny prediction shape = ", outs.shape)
    
    for i in range(outs.shape[0]):
        confidence_on_box = outs[i][4]
        probability_list = outs[i][5:]
        class_index = probability_list.argmax(axis=0)
        probability_on_class = probability_list[class_index]
        score = confidence_on_box * probability_on_class
        
        print(">>> score = ", score)
        
        if (score > 0.3):
            x_center   = outs[i][0] * cols
            y_center   = outs[i][1] * rows
            width      = outs[i][2] * cols
            height     = outs[i][3] * rows
            
            left       = int(x_center - width * 0.5)
            top        = int(y_center - height * 0.5)
            right      = int(x_center + width * 0.5)
            bottom     = int(y_center + height * 0.5)
            
            cv2.rectangle(frame, (left, top), (left + width, top + height), color=(0, 255, 0), thickness=3)
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()