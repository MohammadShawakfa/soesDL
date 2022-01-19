# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 00:27:57 2021

@author: Lenovo
"""
#Import required packages
import face_recognition
import cv2
#image_to_detect = cv2.imread('images/testing/trump-modi.jpg')
#cv2.imshow("test", image_to_detect)

#Loading the image to detect
#original_image = cv2.imread('images/testing/shawakfa_lake.JPG')


#/////////////////////////////////////////
#loading Rania image's to detect
original_image = cv2.imread('images/testing/rania_fr.JPEG')


 
#load the sample images and get the 128 face embeddings from them
moh_image = face_recognition.load_image_file('images/samples/moh_fr.jpeg')
moh_face_encodings = face_recognition.face_encodings(moh_image)[0]

rania_image = face_recognition.load_image_file('images/samples/rania_fr.jpeg')
rania_face_encodings = face_recognition.face_encodings(rania_image)[0]

#save the encodings and the corresponding labels in seperate arrays in the same order
known_face_encodings = [moh_face_encodings,rania_face_encodings]
known_face_names = ["Mohammad Shawakfa","Rania AlOun"]

#load the unkonwn image "Mohammad" to recognize faces in it
#image_to_recognize_moh = face_recognition.load_image_file('images/testing/shawakfa_lake.JPG')


#////////////////////////////////////
#load the unkonwn image "Rania" to recognize faces in it
image_to_recognize_moh = face_recognition.load_image_file('images/testing/rania_fr.JPEG')

#detect all faces in the image
#arguments are image, no_of_times_to_upsample, model
all_face_locations = face_recognition.face_locations(image_to_recognize_moh,model='hog')

#detect face encodings for all the faces detected
#2 parameters , the 1st if the image to analayze the 2nd is all knonwn face locations 
all_face_encodings =  face_recognition.face_encodings(image_to_recognize_moh,all_face_locations)




#cnn model is very accurate but takes a lot of time on the cpu, while hog is a lot faster but
#it's a tradeoff with the acurracy 
#detect all faces in the image (Return the number of faces in the image)
all_face_locations = face_recognition.face_locations(image_to_recognize_moh,model='hog')

#Print the number of faces deteced
print('There are {} face(s) in this image'.format(len(all_face_locations)))

#locate faces location using four points
#looping through the face(s) location(s) and the face embeddings

for current_face_location,current_face_encoding in zip(all_face_locations,all_face_encodings):
    #splitting the tuple to get the four position values
    top_pos,right_pos,bottom_pos,left_pos = current_face_location
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
    cv2.rectangle(original_image,(left_pos,top_pos),(right_pos,bottom_pos),(255,0,0),2)
    
    #display the name as text in the image
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(original_image,name_of_person,(left_pos,bottom_pos  ), font, 0.5,(255,255,255),1)
    
    #display the image
    cv2.imshow("Faces Identified",original_image)
    