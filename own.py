# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 12:14:33 2020

@author: kingslayer
"""

#importing the libraries
import cv2

#import cascadings
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade=cv2.CascadeClassifier('haarcascade_smile.xml')

#Defining the detection function
def detect(gray,frame):
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
        eyes=eye_cascade.detectMultiScale(roi_gray,1.3,5)
        smiles=smile_cascade.detectMultiScale(roi_gray,1.6,15)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        for (fx,fy,fw,fh) in smiles:
            cv2.rectangle(roi_color,(fx,fy),(fx+fw,fy+fh),(0,0,255),2)
            cv2.putText(frame, 'Smiling', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        
    return frame

#Doing face detection
video_capture=cv2.VideoCapture(0)
while True:
    _,frame=video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas=detect(gray,frame)
    cv2.imshow("Video",canvas)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
