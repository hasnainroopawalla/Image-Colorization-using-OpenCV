import numpy as np
import imutils
import cv2

prototxt = "model/colorization_deploy_v2.prototxt"
model = "model/colorization_release_v2.caffemodel"
hull_points = "model/pts_in_hull.npy"
width = 500

# load model
net = cv2.dnn.readNetFromCaffe(prototxt, model)
pts = np.load(hull_points)

# Each point is treated as a 1x1 convolution and passed through the model
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

def processimg(image):
	scaled = image.astype("float32") / 255.0

	# Image is converted to Lab Color format
	lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

	# Caffe Model takes 224x224 image as input
	resized = cv2.resize(lab, (224, 224))

	# Extract L channel (Intensity)
	L = cv2.split(resized)[0] 

	# Mean Subtraction (for Mean Centering)
	L -= 50

	# Network takes L channel as input and predicts the ab channel values
	net.setInput(cv2.dnn.blobFromImage(L))
	ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

	# Resize ab to match dimensions of input image
	ab = cv2.resize(ab, (image.shape[1], image.shape[0]))


	# Concatenate original image L channel value with predicted ab values
	L = cv2.split(lab)[0]
	colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

	# Convert back to RGB
	colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
	colorized = np.clip(colorized, 0, 1)

	colorized = (255 * colorized).astype("uint8")
	
	return colorized