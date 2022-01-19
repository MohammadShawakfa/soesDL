# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 10:27:57 2021

@author: Lenovo
"""

#Import required packages
import face_recognition
import cv2

#Get the webcam #0 (the default one,1,2 etc means additional attached cams)
#This line will return a video stream which is every frame of the video
webcam_video_stream = cv2.VideoCapture(0)
#load the sample images and get the 128 face embeddings from them

moh_image = face_recognition.load_image_file('images/samples/moh_fr.jpeg')
moh_face_encodings = face_recognition.face_encodings(moh_image)[0]

rania_image = face_recognition.load_image_file('images/samples/rania_fr.jpeg')
rania_face_encodings = face_recognition.face_encodings(rania_image)[0]


#save the encodings and the corresponding labels in seperate arrays in the same order
known_face_encodings = [moh_face_encodings,rania_face_encodings]
known_face_names = ["Mohammad Shawakfa","Rania AlOun"]

#initalize the variable to hold all face locations, encodings and names in the frame
all_face_locations =[]
all_face_encodings =[]
all_face_names =[]

#loop through every frame in the video
while True:
    #get the current frame from the video stream as an image
    ret,current_frame = webcam_video_stream.read()
    #Resize the current frame to 1/4 size to process faster with less resources
    current_frame_small = cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)
    #detect all faces in the image (Return the number of faces in the image)
    all_face_locations = face_recognition.face_locations(current_frame_small,number_of_times_to_upsample=2,model='hog')
    #///////////////////////////////////////
    all_face_encodings =  face_recognition.face_encodings(current_frame_small,all_face_locations)


#locate faces location using four points
#looping through the face(s) location(s) and the face embeddings

    for current_face_location,current_face_encoding in zip(all_face_locations,all_face_encodings):
        #splitting the tuple to get the four position values
        top_pos,right_pos,bottom_pos,left_pos = current_face_location
        
        #Change the position magnitude to fit the actual size video frame
        top_pos = top_pos*4
        right_pos = right_pos*4
        bottom_pos = bottom_pos*4
        left_pos = left_pos*4
        #find all the matches and get the list of matches
        all_matches = face_recognition.compare_faces(known_face_encodings,current_face_encoding)
        #string to hold the label
        name_of_person = 'Unkown face'
        #check if all_matches have at leats one item
        #if yes, get the index number of face that is located in the first index of all_matches
        if True in all_matches:
            first_match_index = all_matches.index(True)
            name_of_person = known_face_names[first_match_index]
            
        #draw rectangle around the face
        cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(255,0,0),2)
        
        #display the name as text in the image
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame,name_of_person,(left_pos,bottom_pos  ), font, 0.5,(255,255,255),1)
    
    #display the image
    cv2.imshow("Faces Identified",current_frame)







    #showing the current face with rectangle drawn
    cv2.imshow("Webcam video",current_frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
#Release webcam and close all windows
webcam_video_stream.release()
cv2.destroyAllWindows()
 

 