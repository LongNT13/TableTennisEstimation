import cv2 
import numpy as np 
  
  
# Read in the cascade classifiers for face and eyes 
face_cascade = cv2.CascadeClassifier('model/1/5stages/cascade.xml')
  
  
  
# create a function to detect face 
def adjusted_detect_face(img): 
      
    face_img = img.copy() 
    print("asd")      
    face_rect = face_cascade.detectMultiScale(face_img) 
    print("asd")  
    for (x, y, w, h) in face_rect: 
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 10)
          
    return face_img 
  
# Reading in the image and creating copies 
img = cv2.imread('Cam1/newimage/CoreView_24_27120f01_0601.png')
  
# Detecting the face  
face = adjusted_detect_face(img) 
cv2.imshow("detect", face) 
cv2.waitKey(0)