import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
sys.path.append(str(FILE.parents[1]))
if str(ROOT) not in sys.path:
	sys.path.append(str(ROOT))

import numpy as np
import cv2

from PIL import ImageFilter, ImageStat
from PIL import Image
import os
import time

class ImageQuality():
	def __init__(self):
		self.percentiles = [10, 70]
		self.threshold = {
			"dark": 0.4,
			"light": 0.59,
			"entropy": 0.3,
			"blurry": 0.5
		}
	
	def calc_laplacian_var(self, img):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		lap_value = cv2.Laplacian(img, cv2.CV_16S, ksize=3).var()

		lap_value = 1 - np.exp(-1 * (lap_value * 1e-3))

		return lap_value

	def calc_brightness(self, img):
		image = img.copy()
		if len(image.shape) == 3:
			b, g, r = (
				image[:, :, 0].astype("int"),
				image[:, :, 1].astype("int"),
				image[:, :, 2].astype("int"),
			)
			pixel_brightness = (
			np.sqrt(0.241 * (r * r) + 0.691 * (g * g) + 0.068 * (b * b))
		) / 255
		else:
			pixel_brightness = image / 255.0

		perc_values = np.percentile(pixel_brightness, self.percentiles)

		brightness = 1 - perc_values[0] # percentiles_5
		darkness = perc_values[-1] # percentiles_99

		return brightness, darkness


	def calc_entropy(self, img, normalizing_factor=0.1) -> float:
		image = Image.fromarray(img[..., ::-1])
		entropy = image.entropy()
		assert isinstance(
			entropy, float
		)  # PIL does not have type ann stub so need to assert function return
		scores_data = entropy * normalizing_factor
		scores_data = 1 if scores_data > 1 else scores_data
		return scores_data


	def calc_blurriness(self, img, max_resolution=64, normalizing_factor=0.01, color_threshold=0.18):
		image = Image.fromarray(img[..., ::-1])
		ratio = max(image.width, image.height) / max_resolution
		if ratio > 1:
			resized_image = image.resize(
				(max(int(image.width // ratio), 1), max(int(image.height // ratio), 1))
			)
		else:
			resized_image = image.copy()

		gray_image = resized_image.convert('L')
		edges = gray_image.filter(ImageFilter.FIND_EDGES)
		blurriness = np.sqrt(ImageStat.Stat(edges).var[0])
		gray_std = np.std(gray_image.histogram())

		blur_scores = 1 - np.exp(-1 * blurriness * normalizing_factor)
		std_scores  = 1 - np.exp(-1 * gray_std * normalizing_factor)
		std_scores = 0 if std_scores < color_threshold else std_scores

		score = min(blur_scores + std_scores, 1)

		return score  # type:ignore

	def infer(self, im):
		st_time = time.time()
		brightness, darkness = self.calc_brightness(im)
		blurriness = self.calc_blurriness(im)
		# entropy = calc_entropy(im)
		# lap = calc_laplacian_var(im)

		# print(brightness, darkness, blurriness)
		quality_mes = ""
		quality_mes_en = ""
		if brightness < self.threshold["light"]:
			quality_mes = ". Chú ý: ảnh chụp quá sáng" # ". Warning: the face in the captured photo is too bright"
			quality_mes_en = ". Warning: the face in the captured photo is too bright"
		elif darkness < self.threshold["dark"]:
			quality_mes = ". Chú ý: ảnh chụp quá tối" # ". Warning the face in the captured photo is too dark"
			quality_mes_en = ". Warning the face in the captured photo is too dark"
		elif darkness < self.threshold["blurry"]:
			quality_mes = ". Chú ý: ảnh chụp bị nhiễu" # ". Warning the face in the captured photo is too blurry"
			quality_mes_en = ". Warning the face in the captured photo is too blurry"
		print(f"----Duration quality: {time.time()-st_time}")
		return quality_mes, quality_mes_en
