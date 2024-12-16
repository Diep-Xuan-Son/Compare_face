import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
sys.path.append(str(FILE.parents[1]))
if str(ROOT) not in sys.path:
	sys.path.append(str(ROOT))

from libs.base_libs import *

class GlassesDetection(object):
	def __init__(self, model_path):
		devices = ort.get_available_providers()
		if 'CUDAExecutionProvider' in devices:
			providers = ['CPUExecutionProvider', 'CUDAExecutionProvider']
		else:
			providers = ['CPUExecutionProvider']
		self.sess = ort.InferenceSession(model_path, providers=devices)

	def preprocess(self, img):
		# prep = transforms.Compose([
		# 		transforms.Resize((512, 512)),  # Resize the image to the required input size (512, 512)
		# 		transforms.ToTensor(),
		# 		transforms.Normalize(
		# 			mean=[0.485, 0.456, 0.406],
		# 			std=[0.229, 0.224, 0.225]
		# 			),
		# 		])
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
		img = np.float32(img)
		MEAN = 255 * np.array([[[0.485]], [[0.456]], [[0.406]]])
		STD = 255 * np.array([[[0.229]], [[0.224]], [[0.225]]])
		img = img.transpose(-1, 0, 1)

		img = (img - MEAN) / STD
		input_tensor = img[None].astype(np.float32)
		# input_tensor = torch.from_numpy(img.astype(np.float32))
		# input_tensor = prep(img)
		return input_tensor

	def infer(self, frame, face_bbox):
		def sigmoid(z):
			return 1/(1 + np.exp(-z))
		st_time = time.time()
		frame = frame[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]]
		(h, w) = frame.shape[:2]
		# cv2.imwrite("dfads1.jpg", frame[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]])
		# frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
		input_tensor = self.preprocess(frame)
		ort_inputs = {self.sess.get_inputs()[0].name: input_tensor}
		output = self.sess.run(None, ort_inputs)[0][0,0]

		segmentation_mask = sigmoid(output)
		segmentation_mask = (segmentation_mask > 0.8).astype(np.uint8)*255  # Convert to binary mask
		segmentation_mask = cv2.resize(segmentation_mask, (w,h), interpolation=cv2.INTER_NEAREST)

		frame[segmentation_mask==255] = (255,0,0)
		# cv2.imwrite("dfads1.jpg", frame)

		ratio = segmentation_mask.sum()/255/(w*h)
		is_glasses = False
		# print("----ratio: ", ratio)
		if ratio > 0.12:
			is_glasses = True
		print(f"----Duration glasses: {time.time()-st_time}")
		return is_glasses 
