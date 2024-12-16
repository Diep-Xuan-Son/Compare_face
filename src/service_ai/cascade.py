import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
sys.path.append(str(FILE.parents[1]))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import cv2
import time
import asyncio
import numpy as np

def compute_iou(box, boxes):
	# Compute xmin, ymin, xmax, ymax for both boxes
	xmin = np.maximum(box[0], boxes[:, 0])
	ymin = np.maximum(box[1], boxes[:, 1])
	xmax = np.minimum(box[2], boxes[:, 2])
	ymax = np.minimum(box[3], boxes[:, 3])

	# Compute intersection area
	intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

	# Compute union area
	box_area = (box[2] - box[0]) * (box[3] - box[1])
	boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
	# union_area = box_area + boxes_area - intersection_area
	# print("----intersection_area:  ", intersection_area)
	# print("----box_area:  ", boxes_area)

	# Compute IoU
	iou_box = intersection_area / box_area
	iou_boxes = intersection_area / boxes_area

	iou = np.maximum(iou_box, iou_boxes)

	return iou

def nms(boxes, other_boxes, iou_thres):
	boxes = np.array(boxes)
	other_boxes = np.array(other_boxes)
	# print("-----------------")
	# else:
	#     full_boxes = np.concatenate((boxes, other_boxes), axis=0)
	num_boxes = len(boxes)
	for i, box in enumerate(boxes[::-1]):
		if len(other_boxes) == 0:
			full_boxes = boxes[:num_boxes-i-1]
		else:
			full_boxes = np.concatenate((boxes[:num_boxes-i-1], other_boxes), axis=0)
		# print(len(full_boxes))
		ious = compute_iou(box, full_boxes)
		# print("----ious: ", ious)
		indices = np.where(ious > iou_thres)[0]
		# print("----indices: ", indices)
		if len(indices)!=0:
			# boxes.pop(len(full_boxes)-i-1)
			boxes = np.delete(boxes, num_boxes-i-1, 0)
	# print("/////////////////")
	return boxes
	# num_box = len(full_boxes)
	# temp_boxes = full_boxes.copy()
	# for i in range(num_box):


def xywh2xyxy(xywh):
	xywh = np.array(xywh)
	if len(xywh)==0:
		return []
	else:
		xywh[:, 2:] += xywh[:, :2]
	return xywh

class Cascade():
	def __init__(self, ):
		self.eye_cascade = cv2.CascadeClassifier(f'{str(ROOT)}/cascade_file/haarcascade_eye.xml')
		self.mouth_cascade = cv2.CascadeClassifier(f'{str(ROOT)}/cascade_file/haarcascade_mcs_mouth.xml')
		self.nose_cascade = cv2.CascadeClassifier(f'{str(ROOT)}/cascade_file/haarcascade_mcs_nose.xml')

	async def infer_eye(self, roi_gray):
		eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.3, 18, flags=cv2.CASCADE_FIND_BIGGEST_OBJECT) # 2.0, 18
		eyes_xyxy = xywh2xyxy(eyes)
		eyes = nms(boxes=eyes_xyxy, other_boxes=[], iou_thres=0.85)
		return eyes

	async def infer_nose(self, roi_gray):
		nose = self.nose_cascade.detectMultiScale(roi_gray, 1.3, 18, flags=cv2.CASCADE_FIND_BIGGEST_OBJECT) # 1.3, 18
		nose_xyxy = xywh2xyxy(nose)
		nose = nms(boxes=nose_xyxy, other_boxes=[], iou_thres=0.85)
		return nose

	async def infer_mouth(self, roi_gray):
		mouth = self.mouth_cascade.detectMultiScale(roi_gray, 1.3, 18, flags=cv2.CASCADE_FIND_BIGGEST_OBJECT) # 2.0, 18
		mouth_xyxy = xywh2xyxy(mouth)
		mouth = nms(boxes=mouth_xyxy, other_boxes=[], iou_thres=0.85)
		return mouth

	async def infer(self, croped_img):
		st_time = time.time()
		# print(croped_img.shape)
		croped_img = cv2.resize(croped_img, (int(croped_img.shape[1]*3.75),int(croped_img.shape[0]*5)))
		# cv2.imwrite("abc.jpg", croped_img)
		roi_gray = cv2.cvtColor(croped_img, cv2.COLOR_BGR2GRAY)
		# print(roi_gray.shape)

		results = await asyncio.gather(*[
			self.infer_eye(roi_gray),
			self.infer_nose(roi_gray),
			self.infer_mouth(roi_gray),
		])
		eyes = results[0]
		nose = results[1]
		mouth = results[2]
		
		#print("----eyes: ", eyes)
		#print("----nose: ", nose)
		#print("----mouth: ", mouth)
		check_eye_cover = False
		check_nose_cover = False
		check_mouth_cover = False
		part_cover = ""
		part_cover_en = ""
		if len(eyes)<2:
			check_eye_cover = True
			part_cover += "mắt, "
			part_cover_en += "eyes, "
		if len(nose)<1:
			check_nose_cover = True
			part_cover += "mũi, "
			part_cover_en += "nose, "
		if len(mouth)<1:
			check_mouth_cover = True
			part_cover += "miệng, "
			part_cover_en += "mouth, "
		part_cover = part_cover.rstrip(", ").capitalize()
		print(f"----Duration cascade: {time.time()-st_time}")
		return check_eye_cover, check_nose_cover, check_mouth_cover, part_cover, part_cover_en
