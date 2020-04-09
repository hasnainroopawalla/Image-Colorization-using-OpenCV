from imutils.video import VideoStream
import numpy as np
import process
import imutils
import cv2

input_video = "videos/bw.mp4"
vs = cv2.VideoCapture(input_video)

while True:
	frame = vs.read()[1]

	if frame is None:
		break

	frame = imutils.resize(frame, width=500)
	colorized = process.processimg(frame)
	
	cv2.imshow("Original", frame)
	cv2.imshow("Colored", colorized)

	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

vs.release()

cv2.destroyAllWindows()