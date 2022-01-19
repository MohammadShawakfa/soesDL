# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 00:27:57 2021

@author: Lenovo
"""
#Import required packages
import face_recognition
import cv2
image_to_detect = cv2.imread('images/testing/trump-modi.jpg')
cv2.imshow("test", image_to_detect)

 

#cnn model is very accurate but takes a lot of time on the cpu, while hog is a lot faster but
#it's a tradeoff with the acurracy 
#detect all faces in the image (Return the number of faces in the image)
all_face_locations = face_recognition.face_locations(image_to_detect,model='hog')

#Print the number of faces deteced
print('There are {} face(s) in this image'.format(len(all_face_locations)))

#locate faces location using four points
#looping through the face(s) location(s)

for index,current_face_location in enumerate(all_face_locations):
    #splitting the tuple to get the four position values
    top_pos,right_pos,bottom_pos,left_pos = current_face_location
    #printing the location of current face
    print('Found face {} at top:{},right:{},bottom:{},left:{}'.format(index +1,top_pos,right_pos,bottom_pos,left_pos))
    #Slicing image array by positions inside the loop
    #pay attention that the order in the code is changed
    current_face_image = image_to_detect[top_pos:bottom_pos,left_pos:right_pos]
    #show the sliced images in differnt windows depending on the number of faces i detected
    cv2.imshow("face no" + str(index+1),current_face_image)

    
    





