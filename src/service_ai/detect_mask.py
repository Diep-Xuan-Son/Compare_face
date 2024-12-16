import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
sys.path.append(str(FILE.parents[1]))
if str(ROOT) not in sys.path:
	sys.path.append(str(ROOT))

import threading
import onnxruntime
import numpy as np
import cv2
import time

class BaseOnnx(threading.Thread):
	def __init__(self, path):
		threading.Thread.__init__(self)
		devices = onnxruntime.get_available_providers()
		if 'CUDAExecutionProvider' in devices:
			devices = ['CPUExecutionProvider', 'CUDAExecutionProvider']
		else:
			devices = ['CPUExecutionProvider']
		self.session = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'])

	def pre(self, input_feed):
		for key in input_feed:
			input_feed[key] = input_feed[key].astype(np.float32)
		return input_feed

	def pos(self, out):
		return out[0]
	
	def infer(self, input_feed):
		out = self.session.run(None, input_feed=input_feed)
		return out
	
	def __call__(self, input_feed):
		st_time = time.time()
		input_feed = self.pre(input_feed)
		out = self.infer(input_feed)
		out = self.pos(out)
		print(f"----Duration mask: {time.time()-st_time}")
		return out

class MaskClasifyOnnx(BaseOnnx):
	def __init__(self, path):
		super().__init__(path)
	
	def pre(self, im):
		if isinstance(im, str):
			im = cv2.imread(im)
		im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
		im = cv2.resize(im, (224, 224))
		im = (im / 127.5) - 1.0
		im = np.expand_dims(im, axis=0)
		im = im.astype(np.float32)
		return {'input_1': im}
	
	def pos(self, out):
		(mask, withoutMask) = out[0][0]
		# label = "Mask" if mask > withoutMask else "No Mask"
		is_mask = False
		if mask > withoutMask:
			is_mask = True
		return is_mask
