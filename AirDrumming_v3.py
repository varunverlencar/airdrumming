# import the necessary packages
from collections import deque
import numpy as np
import argparse
import imutils
import cv2
import time
import subprocess
# import matplotlib.pyplot as plt

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=32,
	help="max buffer size")
args = vars(ap.parse_args())

#initialize variables
center = [] # center of circles
wt,ht = 0,0	# width, hieght of frame
X1,Y1,X2,Y2 = [],[],[],[] # x,y coordinates of the drumsticks
t1,t2 = [],[] # time ref for velocities
velX1,velY1,velX2,velY2 = 0,0,0,0 #velocities in x,y
counter1,counter2  = 0,0 #counter for velocity calc
c = 0	# no of hough circles

# loading sounds
# bass_sound = pygame.mixer.Sound('./Sounds/kick-classic.wav')
# crash_sound = pygame.mixer.Sound('./Sounds/crash.wav')
# tom1_sound = pygame.mixer.Sound('./Sounds/tom1.wav')
# tom2_sound = pygame.mixer.Sound('./Sounds/tom2.wav')
# tom3_sound = pygame.mixer.Sound('./Sounds/tom3.wav')
# snare_sound = pygame.mixer.Sound('./Sounds/snare.wav')
# hihat_sound = pygame.mixer.Sound('./Sounds/hihat.wav')

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

	if args.get("video") and not grabbed:
		break

	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame, width=900)
	ht = np.size(frame,0)
	wt = np.size(frame,1)

	#flip frames to match viewer actions
	frame = cv2.flip(frame, 1)

	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	# cv2.imshow("GaussianBlur", blurred)   

    # Convert BGR to HSV
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
	lower_color = np.array([80,60,60])
	upper_color = np.array([170,255,255])

    # define range of green color in HSV
	# lower_color = np.array([29, 86, 6])
	# upper_color = np.array([64, 255, 255])


    # Threshold the HSV image to get only blue colors
	hsvmask = cv2.inRange(hsv, lower_color, upper_color)
	# cv2.imshow("HSV Color Segmentation", hsvmask)

	# perfor erosion and dilation
	ekernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
	emask = cv2.erode(hsvmask, ekernel, iterations=2)
	# cv2.imshow("erosion",emask)

	dkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
	dmask = cv2.dilate(emask, dkernel, iterations=5)
	
	
	# dmask = cv2.flip(dmask, 1)

	#Hough circles	
	circles = cv2.HoughCircles(dmask, cv2.HOUGH_GRADIENT, 1.5, 80,param1=50,param2=22,minRadius=10,maxRadius=45)

	# ensure at least some circles were found
	if circles is not None:
		# convert the (x, y) coordinates and radius of the circles to integers
		circles = np.round(circles[0, :]).astype("int")
		
		# loop over the (x, y) coordinates and radius of the circles
		for (x, y, r) in circles:
			
			#neglect circle with radius smaller than  and store its x,y coordinates
			if circles[0][2] > 15:
				if (circles[0][0] < wt/2):	# assumed drumstick on left 
					X1.append(circles[0][0])
					Y1.append(circles[0][1])
					t1.append(time.time())
					c = 1	# drumstick 1
					counter1 += 1
				else:
					X2.append(circles[0][0])
					Y2.append(circles[0][1])
					t2.append(time.time())
					c = 2  # drumstick 2
					counter2 += 1					
					# print circles[0][1],"c1"

			if len(circles)>1 :
				if circles[1][2] > 15:
					if (circles[1][0] > wt/2):	# assumed drumstick on right 
						X1.append(circles[1][0])
						Y1.append(circles[1][1])
						t1.append(time.time())
						c = 1	# drumstick 1
						counter1 += 1
					else:				
						X2.append(circles[1][0])
						Y2.append(circles[1][1])
						t2.append(time.time())
						c = 2  # drumstick 2
						counter2 += 1
						# print circles[1][1],"c2"

			# draw the circle in the output image,
			# corresponding to the center of the circle	
			cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
			cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

	# calculate the velocities of each drumsticks
	# check to see if enough points have been accumulated in
	# the buffer
	if counter1 >= 10 and  X1[-10] is not None:
		# compute the difference between the x and y
		# coordinates and re-initialize the direction
		velX1 = (X1[-10] - X1[0])/(t1[-10] - t1[0])
		velY1 = (Y1[-10] - Y1[0])/(t1[-10] - t1[0])
		counter1 = 0

	if counter2 >= 10 and  X2[-10] is not None:
		# compute the difference between the x and y
		# coordinates and re-initialize the direction
		velX2 = (X2[-10] - X2[0])/(t2[-10] - t2[0])
		velY2 = (Y2[-10] - Y2[0])/(t2[-10] - t2[0])
		counter2 = 0

	cv2.putText(frame,str('Vel1= '+ str(velX1)+','+str(velY1)), (10*wt/600,420*ht/450), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,255), thickness=1)
	cv2.putText(frame,str('Vel2= ' +str(velX2)+','+str(velY2)), (10*wt/600,436*ht/450), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,255,0), thickness=1)


	scale = 0   #scaling the ROI box
	xout = 0    #manually seperating the ROI in outwards direction along x axis
	yout = 0    #manually seperating the ROI in outwards direction along y axis

	#create ROI and placements for each drum
	#height and width of roi
	box_height,box_width = 60*ht/450,80*wt/600

	# HIHAT 
	xh,yh = int(wt/2 - wt*180/600 - scale) + xout,int(ht/2 - ht*80/450 - scale) + yout
	hihat = frame[yh-box_height:yh,xh-box_width:xh]
	frame = cv2.rectangle(frame,(xh,yh),(xh-box_width,yh-box_height),(0,255,0),3)
	cv2.putText(frame, "HIHAT", (xh-box_width,yh-box_height-5), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,255,0), thickness=1)

	#SNARE
	ysnare,xsnare = yh+100*ht/450 +yout, int(wt/2 - wt*90/600 - scale) + xout
	snare = frame[ysnare-box_height:ysnare,xsnare-box_width:xsnare]
	frame = cv2.rectangle(frame,(xsnare,ysnare),(xsnare-box_width,ysnare-box_height),(0,255,255),3)
	cv2.putText(frame, "SNARE", (xsnare-box_width,ysnare-box_height -5), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,255,255), thickness=1)

	#TOM1
	ytom1,xtom1 = yh+10*ht/450 +yout, int(wt/2 - wt*20/600 - scale) + xout
	tom1 = frame[ytom1-box_height:ytom1,xtom1-box_width:xtom1]
	frame = cv2.rectangle(frame,(xtom1,ytom1),(xtom1-box_width,ytom1-box_height),(255,255,0),3)
	cv2.putText(frame, "TOM1", (xtom1-box_width,ytom1-box_height-5), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,0), thickness=1)

	#TOM2
	ytom2,xtom2 = yh+10*ht/450 +yout, int(wt/2 + wt*20/600 + box_width + scale) + xout
	tom2 = frame[ytom2-box_height:ytom2,xtom2-box_width:xtom2]
	frame = cv2.rectangle(frame,(xtom2,ytom2),(xtom2-box_width,ytom2-box_height),(100,255,0),3)
	cv2.putText(frame, "TOM2", (xtom2-box_width,ytom2-box_height-5), cv2.FONT_HERSHEY_PLAIN, 1.0, (100,255,0), thickness=1)

	#TOM3
	ytom3,xtom3 = yh+100*ht/450 + yout, int(wt/2 + wt*90/600 + box_width + scale) + xout
	tom3 = frame[ytom3-box_height:ytom3,xtom3-box_width:xtom3]
	frame = cv2.rectangle(frame,(xtom3,ytom3),(xtom3-box_width,ytom3-box_height),(80,25,255),3)
	cv2.putText(frame, "TOM3", (xtom3-box_width,ytom3-box_height-5), cv2.FONT_HERSHEY_PLAIN, 1.0, (80,25,255), thickness=1)

	#BASS DRUM
	ybass,xbass = int(ht/2 + ht*180/450 + scale) + yout, int(wt/2 + box_width/2 + scale) + xout
	bass = frame[ybass-box_height:ybass,xbass-box_width:xbass]
	frame = cv2.rectangle(frame,(xbass,ybass),(xbass-box_width,ybass-box_height),(100,0,255),3)
	cv2.putText(frame, "BASS DRUM", (xbass-box_width-15,ybass-box_height-5), cv2.FONT_HERSHEY_PLAIN, 1.0, (100,0,255), thickness=1)

	#CRASH
	ycrash,xcrash = yh + yout,int(wt/2 + wt*180/600 + box_width + scale)  + xout
	crash = frame[ycrash-box_height:ycrash,xcrash-box_width:xcrash]
	frame = cv2.rectangle(frame,(xcrash,ycrash),(xcrash-box_width,ycrash-box_height),(0,0,255),3)
	cv2.putText(frame, "CRASH", (xcrash-box_width,ycrash-box_height-5), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,255), thickness=1)


	drum1 = 'none'
	drum2 = 'none'

	# check for contact
	if circles is not None:
		if (len(Y1) > 0) and (len(X1) > 0):	
			if (Y1[-1] <= yh) and (Y1[-1] >= yh-box_height) and (X1[-1] <= xh) and (X1[-1] >= xh-box_width):
				drum1 = "HIHAT"
				subprocess.Popen(['aplay', './Sounds/hihat.wav'])
			elif (Y1[-1] <= ybass) and (Y1[-1] >= ybass-box_height) and (X1[-1] <= xbass) and (X1[-1] >= xbass-box_width):
				drum1 = "BASS"
				subprocess.Popen(['aplay', './Sounds/kick-classic.wav'])
			elif (Y1[-1] <= ytom1) and (Y1[-1] >= ytom1-box_height) and (X1[-1] <= xtom1) and (X1[-1] >= xtom1-box_width):
				drum1 = "TOM1"
				subprocess.Popen(['aplay', './Sounds/tom1.wav'])
			elif (Y1[-1] <= ytom2) and (Y1[-1] >= ytom2-box_height) and (X1[-1] <= xtom2) and (X1[-1] >= xtom2-box_width):
				drum1 = "TOM2"
				subprocess.Popen(['aplay', './Sounds/tom2.wav'])
			elif (Y1[-1] <= ytom3) and (Y1[-1] >= ytom3-box_height) and (X1[-1] <= xtom3) and (X1[-1] >= xtom3-box_width):
				drum1 = "TOM3"
				subprocess.Popen(['aplay', './Sounds/tom3.wav'])
			elif (Y1[-1] <= ysnare) and (Y1[-1] >= ysnare-box_height) and (X1[-1] <= xsnare) and (X1[-1] >= xsnare-box_width):
				drum1 = "SNARE"
				subprocess.Popen(['aplay', './Sounds/snare.wav'])
			elif (Y1[-1] <= ycrash) and (Y1[-1] >= ycrash-box_height) and (X1[-1] <= xcrash) and (X1[-1] >= xcrash-box_width):
				drum1 = "CRASH"	
				subprocess.Popen(['aplay', './Sounds/crash.wav'])
		
		if (len(Y2) > 0) and (len(X2) > 0):
			if (Y2[-1] <= yh) and (Y2[-1] >= yh-box_height) and (X2[-1] <= xh) and (X2[-1] >= xh-box_width):
				drum2 = "HIHAT"
				subprocess.Popen(['aplay', './Sounds/hihat.wav'])
			elif (Y2[-1] <= ybass) and (Y2[-1] >= ybass-box_height) and (X2[-1] <= xbass) and (X2[-1] >= xbass-box_width):
				drum2 = "BASS"
				subprocess.Popen(['aplay', './Sounds/kick-classic.wav'])
			elif (Y2[-1] <= ysnare) and (Y2[-1] >= ysnare-box_height) and (X2[-1] <= xsnare) and (X2[-1] >= xsnare-box_width):
				drum2 = "SNARE"
				subprocess.Popen(['aplay', './Sounds/snare.wav'])	
			elif (Y2[-1] <= ytom1) and (Y2[-1] >= ytom1-box_height) and (X2[-1] <= xtom1) and (X2[-1] >= xtom1-box_width):
				drum2 = "TOM1"
				subprocess.Popen(['aplay', './Sounds/tom1.wav'])
			elif (Y2[-1] <= ytom2) and (Y2[-1] >= ytom2-box_height) and (X2[-1] <= xtom2) and (X2[-1] >= xtom2-box_width):
				drum2 = "TOM2"
				subprocess.Popen(['aplay', './Sounds/tom2.wav'])
			elif (Y2[-1] <= ytom3) and (Y2[-1] >= ytom3-box_height) and (X2[-1] <= xtom3) and (X2[-1] >= xtom3-box_width):
				drum2 = "TOM3"
				subprocess.Popen(['aplay', './Sounds/tom3.wav'])
			elif (Y2[-1] <= ycrash) and (Y2[-1] >= ycrash-box_height) and (X2[-1] <= xcrash) and (X2[-1] >= xcrash-box_width):
				drum2 = "CRASH"
				subprocess.Popen(['aplay', './Sounds/crash.wav'])
		
			
	#flip the drums
	tom1 = cv2.flip(tom1, 1)
	tom2 = cv2.flip(tom2, 1)
	tom3 = cv2.flip(tom3, 1)
	crash = cv2.flip(crash, 1)
	bass = cv2.flip(bass, 1)
	snare = cv2.flip(snare, 1)
	hihat = cv2.flip(hihat, 1)

	#Title for each processed frame
	cv2.putText(frame, str(drum1 + " " + drum2), (20*wt/600,30*ht/450), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,0,0), thickness=1)		
	cv2.putText(emask, 'Erosion', (wt/2,30*ht/450), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), thickness=1)
	cv2.putText(dmask, 'Dilation', (wt/2,30*ht/450), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), thickness=1)
	cv2.putText(hsvmask, 'HSV Segmentation', (wt/2-10,30*ht/450), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), thickness=1)
	cv2.putText(hsvmask, 'HSV', (wt/2-10,30*ht/450), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), thickness=1)
	cv2.putText(blurred, 'Smoothing', (wt/2,30*ht/450), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), thickness=1)
	cv2.putText(frame, 'Air Drummer', (wt/2 -10,30*ht/450), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), thickness=1)



	# cv2.imshow("Output",np.hstack([hsvmask,emask, dmask]))
	# cv2.imshow("Output1",np.hstack([blurred,hsv, frame]))
	cv2.imshow("Air Drummer", frame)
	# cv2.imshow("tom1", tom1)
	# cv2.imshow("tom2", tom2)
	# cv2.imshow("tom3", tom3)
	# cv2.imshow("ride", ride)
	# cv2.imshow("bass", bass)
	# cv2.imshow("snare", hihat)
	# cv2.imshow("hihat", snare)

	key = cv2.waitKey(1) & 0xFF
	
	#normalize time
	if len(t1) > 0 :
		for i in np.arange(0,len(t1)-1):
			t1[i] = t1[i] - t1[0]	
	if len(t2) > 0 :
		for i in np.arange(1,len(t2)-1):
			t2[i] = t2[i] - t2[0]

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):

		# plt.subplot(2, 1, 1)
		# ply.plot(Y1,t1)
		# plt.plot(Y2,t2)
		# plt.xlabel('time')
		# plt.ylabel('Y coordinate')
		# plt.title('Plot of y coordinate vs time')
		# plt.legend(['Drumstuck1', 'Drumstick2'])


		# plt.subplot(2, 1, 2)
		# ply.plot(X1,t1)
		# plt.plot(X2,t2)
		# plt.xlabel('time')
		# plt.ylabel('X coordinate')
		# plt.title('Plot of x coordinate vs time')
		# plt.legend(['Drumstuck1', 'Drumstick2'])

		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()