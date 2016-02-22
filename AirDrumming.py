# USAGE
# python ball_tracking.py --video ball_tracking_example.mp4
# python ball_tracking.py

# import the necessary packages
from collections import deque
import numpy as np
import argparse
import imutils
import cv2
import time
# import billiard as multiprocessing
import subprocess
# import playwav
# import pyaudio
# multiprocessing.forking_enable(False)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=5,
	help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
lower_blue = (29, 86, 6)
upper_blue = (64, 255, 255)
# greenLower = (100, 31, 0)
# greenUpper = (255, 160, 160)

# loading sounds
# bass_sound = pygame.mixer.Sound('./Sounds/kick-classic.wav')
# crash_sound = pygame.mixer.Sound('./Sounds/crash.wav')
# tom1_sound = pygame.mixer.Sound('./Sounds/tom1.wav')
# tom2_sound = pygame.mixer.Sound('./Sounds/tom2.wav')
# tom3_sound = pygame.mixer.Sound('./Sounds/tom3.wav')
# snare_sound = pygame.mixer.Sound('./Sounds/snare.wav')
# hihat_sound = pygame.mixer.Sound('./Sounds/hihat.wav')

pts = deque(maxlen=args["buffer"])
center = []
areaArray = []
wt = 0
ht = 0
x = 0
y = 0

vel = np.array([0,0])
velx = 0
vely = 0

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	camera = cv2.VideoCapture(0)

# otherwise, grab a reference to the video file
else:
	camera = cv2.VideoCapture(args["video"])

# keep looping
while True:
	# grab the current frame
	starttime = time.clock()
	(grabbed, frame) = camera.read()
	# template = cv2.imread('drumstick.jpg',0)
	# w, h = template.shape[::-1]

	X = []
	Y = []
	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if args.get("video") and not grabbed:
		break

	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame, width=600)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	    
    	# Convert BGR to HSV
    	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    	# define range of blue color in HSV
    	# lower_blue = np.array([100,40,40])
    	# upper_blue = np.array([170,255,255])

    	
    	# Threshold the HSV image to get only blue colors
    	mask = cv2.inRange(hsv, lower_blue, upper_blue)


    	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    	mask = cv2.erode(mask, kernel, iterations=2)
    	mask = cv2.dilate(mask, kernel, iterations=2)

    	# Bitwise-AND mask and original image
    	#res = cv2.bitwise_and(frame,frame, mask= mask)

    
    	# Setup SimpleBlobDetector parameters.
    	params = cv2.SimpleBlobDetector_Params()

    	# Filter by Area.
    	params.filterByArea = True
    	#params.minArea = 1000000
    	params.maxArea = 10

    	# Create a detector with the parameters
    	detector = cv2.SimpleBlobDetector_create(params)

    	# Set up the detector with default parameters.
    	# detector = cv2.SimpleBlobDetector()
 
    	# Detect blobs.
    	keypoints = detector.detect(mask)
 
    	# Draw detected blobs as red circles.
    	# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    	im_with_keypoints = cv2.drawKeypoints(mask, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 



    	# find contours in the mask and initialize the current (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
	center = None
	ht = np.size(frame,0)
	wt = np.size(frame,1)
	# blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	# hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	# # img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# # res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
	# # threshold = 0.8
	# # loc = np.where( res >= threshold)


	# # construct a mask for the color "green", then perform
	# # a series of dilations and erosions to remove any small
	# # blobs left in the mask
	# mask = cv2.inRange(hsv, greenLower, greenUpper)
	# mask = cv2.erode(mask, np.ones((5,5),np.uint8), iterations=2)
	# mask = cv2.dilate(mask, np.ones((5,5),np.uint8), iterations=2)

	
	# # find contours in the mask and initialize the current
	# # (x, y) center of the ball
	# cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
	# 	cv2.CHAIN_APPROX_SIMPLE)[-2]
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:2]

	# only proceed if at least one contour was found
	if len(cnts) > 0:
		

		for i in range(len(cnts)):
			#find the nth largest contour [n-1][1], in this case 2
			
			# if cv2.contourArea(cnts[i]) < 300:
  	# 			continue
			# find the contours in the mask, then use
			# it to compute the minimum enclosing circle and
			# centroid
			c = cnts[i] #max(cnts, key=cv2.contourArea)
			((x, y), radius) = cv2.minEnclosingCircle(c)
			X.append(int(x))
			Y.append(int(y))

			# print(i)
			# print(x)
			M = cv2.moments(c)
			center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		# only proceed if the radius meets a minimum size
			if radius > 15:
				# draw the circle and centroid on the frame,
				# then update the list of tracked points
				cv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)
				cv2.circle(frame, center, 2, (0, 0, 255), -1)

	# update the points queue
	# pts.appendleft(center)
	
	# pos = np.matrix([X,Y])

	#find x and y velocities
	velx = int(x) - velx
	vely = int(y) - vely
	tstop = time.clock() - starttime
	vel = np.array([velx,vely])
	#print(vel)

	#create ROI and placements for each drumset
	# change positions
	rh, hh,ch,wh = int(ht*3/7),50,int(wt/5),100
	hihat = frame[rh-hh:rh,ch-wh:ch]
	frame = cv2.rectangle(frame,(ch,rh),(ch-wh,rh-hh),(0,255,0),1)

	rsnare, hsnare,csnare,wsnare = rh+80,50,ch+100,100
	snare = frame[rsnare-hsnare:rsnare,csnare-wsnare:csnare]
	frame = cv2.rectangle(frame,(csnare,rsnare),(csnare-wsnare,rsnare-hsnare),(0,255,255),1)

	rtom1, htom1,ctom1,wtom1 = rh,50,ch+170,100
	tom1 = frame[rtom1-htom1:rtom1,ctom1-wtom1:ctom1]
	frame = cv2.rectangle(frame,(ctom1,rtom1),(ctom1-wtom1,rtom1-htom1),(255,255,0),1)

	rtom2, htom2,ctom2,wtom2 = rh,50,ch+275,100
	tom2 = frame[rtom2-htom2:rtom2,ctom2-wtom2:ctom2]
	frame = cv2.rectangle(frame,(ctom2,rtom2),(ctom2-wtom2,rtom2-htom2),(100,255,0),1)

	rtom3, htom3,ctom3,wtom3 = rh+80,50,ch+375,100
	tom3 = frame[rtom3-htom3:rtom3,ctom3-wtom3:ctom3]
	frame = cv2.rectangle(frame,(ctom3,rtom3),(ctom3-wtom3,rtom3-htom3),(80,25,255),1)

	rbass, hbass,cbass,wbass = ht*9/10,50,ch+170,50
	bass = frame[rbass-hbass:rbass,cbass-wbass:cbass]
	frame = cv2.rectangle(frame,(cbass,rbass),(cbass-wbass,rbass-hbass),(100,0,255),1)

	rcrash, hcrash,ccrash,wcrash = rh,50,ch+445,100
	crash = frame[rcrash-hcrash:rcrash,ccrash-wcrash:ccrash]
	frame = cv2.rectangle(frame,(ccrash,rcrash),(ccrash-wcrash,rcrash-hcrash),(0,0,255),1)
	


	#flip the frames
	frame = cv2.flip(frame, 1)
	tom1 = cv2.flip(tom1, 1)
	tom2 = cv2.flip(tom2, 1)
	tom3 = cv2.flip(tom3, 1)
	crash = cv2.flip(crash, 1)
	bass = cv2.flip(bass, 1)
	snare = cv2.flip(snare, 1)
	hihat = cv2.flip(hihat, 1)


   	cv2.putText(frame, str(vel), (20,20), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,0,0), thickness=1)
    
	# show the frame to our screen
	cv2.imshow("mask", mask)
	cv2.imshow("Frame", frame)	
	# cv2.imshow("tom1", tom1)
	# cv2.imshow("tom2", tom2)
	# cv2.imshow("tom3", tom3)
	# cv2.imshow("ride", ride)
	# cv2.imshow("bass", bass)
	# cv2.imshow("snare", hihat)
	# cv2.imshow("hihat", snare)

	# check for contact
	if len(cnts) > 0 and len(X) > 1:
		#crash sound if position is in roi
		if X[0] <= rh and X[0] >= rh-hh and Y[0] <= ch and Y[0] >= ch-wh:
			# afile = './Sounds/kick-classic.wav'
			# if __name__=="__main__" :
   #  			p =  multiprocessing.Process(target=playwav.worker, args = (afile,))
   #  			p.start()
   #  			time.sleep(0.5)
   #  			p.join()
			# crash_sound.play()
			crash_play = subprocess.Popen(['aplay', './Sounds/crash.wav'])
	   		time.sleep(0.2)

	   	#crash sound 
	   	elif  X[1] <= rh and X[1] >= rh-hh and Y[1] <= ch and Y[1] >= ch-wh:
	   		# crash_sound.play()

	   		crash_play = subprocess.Popen(['aplay', './Sounds/crash.wav'])
	   		time.sleep(0.2)

	   	#bass
	   	elif X[0] <= rbass and X[0] >= rbass-hbass and Y[0] <= cbass and Y[0] >= cbass-wbass:
			# bass_sound.play()

			bass_play = subprocess.Popen(['aplay', './Sounds/kick-classic.wav'])
	   		time.sleep(0.2)

	   	elif X[1] <= rbass and X[1] >= rbass-hbass and Y[1] <= cbass and Y[0] >= cbass-wbass:
			# bass_sound.play()

	   		bass_play = subprocess.Popen(['aplay', './Sounds/kick-classic.wav'])
	   		time.sleep(0.2)

	   	elif X[0] <= rtom1 and X[0] >= rtom1-htom1 and Y[0] <= ctom1 and Y[0] >= ctom1-wtom1:

			tom1_play = subprocess.Popen(['aplay', './Sounds/tom1.wav'])
	   		time.sleep(0.2)


		elif X[1] <= rtom1 and X[1] >= rtom1-htom1 and Y[1] <= ctom1 and Y[1] >= ctom1-wtom1:

			tom1_play = subprocess.Popen(['aplay', './Sounds/tom1.wav'])
	   		time.sleep(0.2)

	   	elif X[0] <= rtom2 and X[0] >= rtom2-htom2 and Y[0] <= ctom2 and Y[0] >= ctom2-wtom2:

			tom2_play = subprocess.Popen(['aplay', './Sounds/tom2.wav'])
	   		time.sleep(0.2)

	   	elif X[1] <= rtom2 and X[1] >= rtom2-htom2 and Y[1] <= ctom2 and Y[1] >= ctom2-wtom2:

			tom2_play = subprocess.Popen(['aplay', './Sounds/tom2.wav'])
	   		time.sleep(0.2)	 

	   	elif X[0] <= rtom3 and X[0] >= rtom3-htom3 and Y[0] <= ctom3 and Y[0] >= ctom3-wtom3:

			tom3_play = subprocess.Popen(['aplay', './Sounds/tom3.wav'])
	   		time.sleep(0.2)  

	   	elif X[1] <= rtom3 and X[1] >= rtom3-htom3 and Y[1] <= ctom3 and Y[1] >= ctom3-wtom3:

			tom3_play = subprocess.Popen(['aplay', './Sounds/tom3.wav'])
	   		time.sleep(0.2)

	   	elif X[0] <= rsnare and X[0] >= rsnare-hsnare and Y[0] <= csnare and Y[0] >= csnare-wsnare:

			snare_play = subprocess.Popen(['aplay', './Sounds/snare.wav'])
	   		time.sleep(0.2)	

	 	elif X[1] <= rsnare and X[1] >= rsnare-hsnare and Y[1] <= csnare and Y[1] >= csnare-wsnare:

			snare_play = subprocess.Popen(['aplay', './Sounds/snare.wav'])
	   		time.sleep(0.2)	 

	   	elif X[0] <= rcrash and X[0] >= rcrash and Y[0] <= ccrash and Y[0] >= ccrash-wcrash:

			crash_play = subprocess.Popen(['aplay', './Sounds/hihat.wav'])
	   		time.sleep(0.2)	 	

	   	elif X[1] <= rcrash and X[1] >= rcrash and Y[1] <= ccrash and Y[1] >= ccrash-wcrash:

			crash_play = subprocess.Popen(['aplay', './Sounds/hihat.wav'])
	   		time.sleep(0.2)	  


	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
