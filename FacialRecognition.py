#Facial Recognition Program for Security Systems
#Program captures a image of the user from a webcam and prompts the user to enter the USER ID.
#The captured image is compared with the database which holds the set of images who can gain access
#If the captured image matches the database, access is granted to the user else access is denied.

#!/usr/bin/python
import cv2,os,getopt
import sys
from cv2 import *
import numpy as np
from PIL import Image
import RPi.GPIO as GPIO ## Import GPIO library
import time
import getpass
GPIO.setmode(GPIO.BOARD)
randval=0
try:
	opts,args = getopt.getopt(sys.argv[1:],":h")
except getopt.GetoptError as err:
	print str(err)
	print "Exiting the program. Can support only -h(help) information"
	sys.exit()

if str(len(opts))== '1':
	for opt,arg in opts:
		if opt =='-h':
			print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
			print "Facial Recognition Program for Security Systems"
			print "Program captures a image of the user from a webcam and prompts the user to enter the USER ID."
			print "The captured image is compared with the database which holds the set of images who can gain access"
			print "If the captured image matches the database, access is granted to the user else access is denied."
			print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
			opt_input = raw_input("Enter any key to continue running the program. Enter 'exit' to terminate the program.(case sensitive):")
			word_opt = "exit"
			if opt_input == word_opt:
				print "\n"
				print "Exiting the program. Thank you."
				sys.exit()
		else:
			print "Invalid Argument. Only '-h'(help) information provided."
			sys.exit()
#start_time = time.time()
#Load a cascade file for detecting faces
face_cascade = cv2.CascadeClassifier('/home/pi/opencv/haarcascade_frontalface_default.xml')

#num_input = raw_input("Enter User ID (If ID within 1-9, prefix zero with ID):"  % getpass.getpass())
num_input= getpass.getpass(prompt = "Enter User ID (If ID within 1-9, prefix zero with ID):")  #User enters the ID which cannot be seen on the terminal
if num_input.isdigit():     #Check if entered User ID is a number, If not prompt user to enter User ID again
	subject_input = 'subject'   #String Manipulation of User Input to obtain a format as subjectUserID.test
	subject_input+= num_input
	subject_input+= '.test'
else:
	print "Enter valid user ID, User ID consists only of numbers."  #Error Message, Prompting the user to enter ID again
	num_input= getpass.getpass(prompt = "Enter User ID again(If ID within 1-9, prefix zero with ID):")
	if num_input.isdigit():
		subject_input = 'subject'
		subject_input+= num_input
		subject_input+= '.test'
	else:
		print "Not a valid User ID. Exiting the program." #Exiting the program if the entered User ID is not a number.
		sys.exit()

video_capture = cv2.VideoCapture(0)   #webcam video capture frame by frame
i=0;
while (i<50):		
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    # Display the resulting frame
    cv2.imshow('Video', frame) #display the video back to the user
    if cv2.waitKey(1) & 0xFF == ord('q'):
       cv2.destroyAllWindows()
       break
    i=i+1;	
    if i==50:
	cv2.imwrite("subject94.jpg",frame) #save image
	img=frame
	os.rename("subject94.jpg",subject_input) #rename the image to desired format
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

if ret:    #frame captured without any errors
     test_var = raw_input("Enter the word 'capture' to capture the image and detect a face (case sensitive):") #sanity check to make sure User is not a robot
     destroyWindow("Captured Image")
     

#Load an image from file
word_compare='capture' #keywork "Capture" comparision
if test_var!=word_compare :
	print "Keyword 'capture' not entered, Exiting the program"
	sys.exit() #exiting the program, if entered keywork does not match
image = img
#imshow("Captured",img)
image = cv2.imread(subject_input, 1) #reading from the captured image
cv2.destroyAllWindows()
#Load a cascade file for detecting faces
face_cascade = cv2.CascadeClassifier('/home/pi/opencv/haarcascade_frontalface_default.xml')  #face detection Haar Cascade XML file

#Convert to grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #Haar Cascade works best with gray scale image and hence the conversion to gray scale

#Look for faces in the image using the loaded cascade file
faces = face_cascade.detectMultiScale(gray, 1.1, 5) #Detection of faces in the captured image
#print "Detected "+str(len(faces))+" faces"
if str(len(faces)) == '0' :  #checking for the number of faces. If no face found, we exit the program
	print "No faces found in the captured image. Exiting the program."
	sys.exit()
if str(len(faces)) > '1' : #if more than one face found, we exit the program
	print "More than '1' face detected in the captured image. Exiting the program."
	sys.exit()

print "Detected "+str(len(faces))+" face. Proceeding to Verification of Database..." #output the number of faces

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

recognizer = cv2.createLBPHFaceRecognizer()  #using a recognition algorithm = LBPH Face Recognition algorithm

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)] #extract path to all the images in the database
    images = [] #create an empty array of images and labels
    labels = []
    print "Verifying Database for Facial Recognition..."
    for image_path in image_paths:
        image_pil = Image.open(image_path).convert('L') #grayscale conversion of all the images in database
        image = np.array(image_pil, 'uint8') #numpy array conversion
        
    # Get the label of the image
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))  #spiltting the name of the image to extract label/ID
        images.append(image) #appending every image to images array
        labels.append(nbr) #appending labels
	#cv2.imshow("Adding faces to traning set...", image)
        #cv2.waitKey(1)
   
   # return the images list and labels list
    return images, labels

# Path to the Yale Dataset(Cropped)
path = '/home/pi/opencv/face_recog/face_recognizer/cropped_yalefaces_database'  

images, labels = get_images_and_labels(path) #function call
cv2.destroyAllWindows()
print "Verification Complete. Proceeding to Training the database..."

#print ("Number of seconds = %s" %(time.time() - start_time))

#training the face detection algorithm using database images
recognizer.train(images, np.array(labels))

#Prediction Algorithm.
image_paths = "/home/pi/opencv/face_recog/face_recognizer/"
image_paths+= subject_input #Path to the captured image


print "Recognition Algorithm Running..."
predict_image_pil = Image.open(image_paths).convert('L') #conversion of grayscale
predict_image = np.array(predict_image_pil, 'uint8') #numpy array conversion

faces = faceCascade.detectMultiScale(predict_image) #Find face in image again(double checking)

for (x, y, w, h) in faces:
    nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])  #predict the label and confidence parameters using predict()
    nbr_actual = int(os.path.split(image_paths)[1].split(".")[0].replace("subject", "")) #extract the actual label
    #print "Entered User ID:{}".format(nbr_actual)
    #print "Predicted User ID:{}".format(nbr_predicted)
    #print "Confidence Value : {}".format(conf)
    if nbr_actual == nbr_predicted: #checking for face recognition
            count=1 #set count =1 if the face matches the database
    else:
	    count = 0 #set count =0 if the face does not match the database
#print ("Number of seconds = %s" %(time.time() - start_time))

if count == 1:
		print "Entered User ID is matching the database. Access Granted"
	 	lvu=11 #green
elif count == 0 :
		print "Entered User ID is not matching the database. Access Denied !"
		lvu=13 #red

speed=10
#GPIO.setmode(GPIO.BOARD) ## Use board pin numbering
GPIO.setup(lvu, GPIO.OUT) ## Setup GPIO Pin 13 to OUT

##Define a function named Blink()
def Blink(speed):

	GPIO.output(lvu,True)## Switch on pin 11
	time.sleep(speed)## Wait

## Start Blink() function. Convert user input from strings to numeric data types and pass to Blink() as parameters
Blink(float(speed))
print "Program Completed. Thank you." ## When loop is complete, print "Done
GPIO.cleanup()
print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

os.remove(subject_input) #delete the captured image after all the processing has been done.
