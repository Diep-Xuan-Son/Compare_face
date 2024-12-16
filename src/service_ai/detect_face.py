import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
sys.path.append(str(FILE.parents[1]))
if str(ROOT) not in sys.path:
	sys.path.append(str(ROOT))
DIR = FILE.parents[1]

import numpy as np
import cv2


class DetectFace():
	def __init__(self, model_path, proto_path):
		# prototxtPath = f"{str(ROOT)}/weights/deploy.prototxt"
		# weightsPath = f"{str(ROOT)}/weights/res10_300x300_ssd_iter_140000.caffemodel"
		self.faceNet = cv2.dnn.readNet(proto_path, model_path)

	async def infer(self, frame):
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
			(104.0, 177.0, 123.0))
		self.faceNet.setInput(blob) # pass the blob through the network and obtain the face detections
		detections = self.faceNet.forward()
		locs = []
		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with
			# the detection
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by ensuring the confidence is
			# greater than the minimum confidence
			if confidence > 0.5:
				# compute the (x, y)-coordinates of the bounding box for
				# the object
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# ensure the bounding boxes fall within the dimensions of
				# the frame
				(startX, startY) = (max(0, startX), max(0, startY))
				(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
				locs.append((startX, startY, endX, endY))
		
		# get the biggest face 
		locs_max = max(locs, key=lambda x: (x[2]-x[0]) * ([x[3]-x[1]])) if len(locs) else []
		# startX, startY, endX, endY = locs_max if len(locs_max) else (0, 0, w, h)
		croped_image = []
		if len(locs_max):
			croped_image = frame[locs_max[1]:locs_max[3], locs_max[0]:locs_max[2]]
		croped_image = np.expand_dims(croped_image, 0)
		return np.array(croped_image), np.array(locs_max)

if __name__=="__main__":
	model_path = f"{str(DIR)}/weights/deploy.prototxt"
	proto_path = f"{str(DIR)}/weights/res10_300x300_ssd_iter_140000.caffemodel"
	dtf = DetectFace(model_path, proto_path)
	img = cv2.imread("test2.jpg")
	croped_image, locs_max = dtf.infer(img)
	cv2.imwrite("aaa.jpg", croped_image[0])