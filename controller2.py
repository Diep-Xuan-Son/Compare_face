import os, sys
from pathlib import Path
import json 
import numpy as np
import cv2
from io import BytesIO
import shutil
import threading
import uvicorn
# import redis
import time
from anyio import to_thread
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Depends, Body
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from schemes import *
from triton_services import *
from libs.utils import *
from multi_worker import myapp

from service_ai.spoof_detection import SpoofDetectionRunnable
from service_ai.spoof_detection_onnx import FakeFace
from service_ai.cascade import Cascade
from service_ai.detect_mask import MaskClasifyOnnx
from service_ai.detect_glasses import GlassesDetection
from service_ai.check_quality import ImageQuality

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

# # SPOOFINGDET = SpoofDetectionRunnable(**{"model_path": f"{str(ROOT)}/weights/spoofing.pt",
# # 									"imgsz": 448,
# # 									"device": 'cpu',
# # 									"cls_names": ['authentic', 'fake']})
# SPOOFINGDET = FakeFace(f"{str(ROOT)}/weights/spoofing.onnx")
# CCA = Cascade()
# MASKCLASIFY = MaskClasifyOnnx(f"{str(ROOT)}/weights/mask_detector.onnx")
# GLASSESDET = GlassesDetection(f"{str(ROOT)}/weights/segmentation_full_lraspp_mobilenet_v3_large.pth")
# IMGQUALITY = ImageQuality()

# TRITONSERVER_IP = os.getenv('TRITONSERVER_IP', "192.168.6.163")
# TRITONSERVER_PORT = os.getenv('TRITONSERVER_PORT', 8001)
# # REDISSERVER_IP = os.getenv('REDISSERVER_IP', "192.168.6.142")
# # REDISSERVER_PORT = os.getenv('REDISSERVER_PORT', 6400)
# print("----TRITONSERVER_IP: ", TRITONSERVER_IP)
# print("----TRITONSERVER_PORT: ", TRITONSERVER_PORT)
# # print("----REDISSERVER_IP: ", REDISSERVER_IP)
# # print("----REDISSERVER_PORT: ", REDISSERVER_PORT)
# tritonClient = get_triton_client(ip_address=f"{TRITONSERVER_IP}:{TRITONSERVER_PORT}")
# # redisClient = redis.StrictRedis(host=REDISSERVER_IP,
# # 								port=int(REDISSERVER_PORT),
# # 								password="RedisAuth",
# # 								db=0)

# class CompareFace(BaseWorker):
# 	def __init__(self):
# 		super().__init__()
# 		self.spoofingdet = FakeFace(f"{str(ROOT)}/weights/spoofing.onnx")
# 		self.cca = Cascade()
# 		self.maskclasify = MaskClasifyOnnx(f"{str(ROOT)}/weights/mask_detector.onnx")
# 		# self.glassesdet = GlassesDetection(f"{str(ROOT)}/weights/segmentation_full_lraspp_mobilenet_v3_large.pth")
# 		self.imgquality = ImageQuality()

# 		TRITONSERVER_IP = os.getenv('TRITONSERVER_IP', "192.168.6.163")
# 		TRITONSERVER_PORT = os.getenv('TRITONSERVER_PORT', 8001)
# 		self.tritonClient = get_triton_client(ip_address=f"{TRITONSERVER_IP}:{TRITONSERVER_PORT}")

# 	def execute(self, proid=0, img_face=None, img_id=None):
# 		print("aaaaa")
# 		time.sleep(5)
# 		print("bbbbb")
# 		t_det = time.time()
# 		#---------------------------face det-------------------------
# 		in_retinaface, out_retinaface = get_io_retinaface(img_face)
# 		results = self.tritonClient.infer(model_name="detection_retinaface_ensemble", inputs=in_retinaface)
# 		croped_image_face = results.as_numpy("croped_image")
# 		if len(croped_image_face)==0:
# 			return {"success": False, "error_code": 8001, "error": "Don't find any face in face photo"}
# 		box_face = results.as_numpy("box")[0]
# 		box_face = box_face.astype(int)
# 		w_crop = (box_face[2]-box_face[0])
# 		h_crop = (box_face[3]-box_face[1])

# 		in_retinaface, out_retinaface = get_io_retinaface(img_id)
# 		results = self.tritonClient.infer(model_name="detection_retinaface_ensemble", inputs=in_retinaface)
# 		croped_image_id = results.as_numpy("croped_image")
# 		if len(croped_image_id)==0:
# 			return {"success": False, "error_code": 8001, "error": "Don't find any face in identification photo"}
# 		#-----------image quality------------
# 		quality_mes = self.imgquality.infer(img_face[box_face[1]:box_face[3], box_face[0]:box_face[2]])
# 		print("----quality_mes: ", quality_mes)
# 		#/////////////////////////////////////
# 		#---------------spoofing--------------
# 		box_expand = np.array([max(box_face[0]-w_crop,0), max(box_face[1]-h_crop,0), min(box_face[2]+w_crop, img_face.shape[1]), min(box_face[3]+h_crop, img_face.shape[0])], dtype=int)
# 		result = self.spoofingdet.inference([img_face[box_expand[1]:box_expand[3], box_expand[0]:box_expand[2]]])[0]
# 		print("---------result_spoofing", result)
# 		if result[1] > 0.85:
# 			# img_list = os.listdir("./image_test")
# 			# cv2.imwrite(f"./image_test/{len(img_list)}.jpg", img_spoofing)
# 			return {"success": False, "error_code": 8002, "error": "Fake face image"+quality_mes}
# 		#//////////////////////////////////////

# 		#---------------glasses----------------
# 		# is_glasses = self.glassesdet.infer(img_face, box_face.astype(int))
# 		# if is_glasses:
# 		# 	return {"success": False, "error_code": 8005, "error": f"Face is convered up by glasses!"+quality_mes}
# 		#//////////////////////////////////////

# 		#------------------mask----------------
# 		is_mask = self.maskclasify(img_face[box_face[1]:box_face[3], box_face[0]:box_face[2]])
# 		if is_mask:
# 			return {"success": False, "error_code": 8006, "error": f"Face is convered up by mask!"+quality_mes}
# 		#//////////////////////////////////////

# 		#---------------cascade----------------
# 		# print(croped_image_face.shape)
# 		# box_expand = np.array([max(box_face[0]-w_crop/4,0), max(box_face[1]-h_crop/4,0), min(box_face[2]+w_crop/4, img_face.shape[1]), min(box_face[3]+h_crop/4, img_face.shape[0])], dtype=int)
# 		check_eye_cover, check_nose_cover, check_mouth_cover, part_cover = self.cca.infer(croped_image_face[0])
# 		if check_eye_cover or check_nose_cover or check_mouth_cover:
# 			return {"success": False, "error_code": 8004, "error": f"{part_cover} was covered up"+quality_mes}
# 		#//////////////////////////////////////

# 		#////////////////////////////////////////////////////////////
# 		print("------Duration det: ", time.time()-t_det)
		
# 		t_reg = time.time()
# 		#---------------------------face reg-------------------------
# 		in_ghostface, out_ghostface = get_io_ghostface(croped_image_face)
# 		results = self.tritonClient.infer(model_name="ghost_face_nodet_ensemble", inputs=in_ghostface, outputs=out_ghostface)
# 		feature_face = results.as_numpy("feature_norm")
# 		feature_face = feature_face.astype(np.float16)

# 		in_ghostface, out_ghostface = get_io_ghostface(croped_image_id)
# 		results = self.tritonClient.infer(model_name="ghost_face_nodet_ensemble", inputs=in_ghostface, outputs=out_ghostface)
# 		feature_id = results.as_numpy("feature_norm")
# 		feature_id = feature_id.astype(np.float16)
# 		#////////////////////////////////////////////////////////////
# 		print("------Duration reg: ", time.time()-t_reg)

# 		t_comp = time.time()
# 		#---------------------------compare face----------------------
# 		print(feature_face.shape)
# 		print(feature_id.shape)
# 		dist = np.linalg.norm(feature_face - feature_id, axis=1)
# 		similarity = (np.tanh((1.23132175 - dist) * 6.602259425) + 1) / 2

# 		print("---------similarity: ", similarity)

# 		if similarity < 0.75:
# 			return {"success": False, "error_code": 8003, "error": "Face photo and identification photo is not similar"+quality_mes}
# 		return {"success": True, "similarity": float(similarity), "warning": quality_mes}\
# 		#/////////////////////////////////////////////////////////////

# # cpf = CompareFace()
# # print(cpf.index)

NUM_WORKER = 2
# myapp = MultiWorker(myclass=getattr(__import__(__name__), "CompareFace"), config={})
# myapp.init_worker(num_worker=NUM_WORKER)

@asynccontextmanager
async def lifespan(app: FastAPI):
	limiter = to_thread.current_default_thread_limiter()
	limiter.total_tokens = NUM_WORKER*6
	yield

app = FastAPI(lifespan=lifespan)
# app.mount("/static", StaticFiles(directory="static"), name="static")

# @app.on_event("startup")
# async def startup_event():
# 	limiter = to_thread.current_default_thread_limiter()
# 	limiter.total_tokens = NUM_WORKER*6
# 	return

@app.post('/set_available_threads')
async def set_available_threads(num_thread: int):
	limiter = to_thread.current_default_thread_limiter()
	limiter.total_tokens = num_thread
	return

@app.get('/get_available_threads')
async def get_available_threads():
	return to_thread.current_default_thread_limiter().available_tokens

@app.post("/api/compareFace")
def compareFace(image_face: UploadFile = File(...), image_identification: UploadFile = File(...)):
	image_byte = image_face.file.read()
	nparr = np.fromstring(image_byte, np.uint8)
	img_face = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

	image_byte = image_identification.file.read()
	nparr = np.fromstring(image_byte, np.uint8)
	img_id = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
	t_det = time.time()
	
	proc_name =  myapp.run(img_face=img_face, img_id=img_id)
	result = myapp.get_result(proc_name=proc_name["nworker"])
	return result

# @app.post("/api/spoofingCheck")
# async def spoofingCheck(image: UploadFile = File(...)):
# 	try:
# 		image_byte = await image.read()
# 		nparr = np.fromstring(image_byte, np.uint8)
# 		img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
# 		#---------------------------face det-------------------------
# 		in_retinaface, out_retinaface = get_io_retinaface(img)
# 		results = await tritonClient.infer(model_name="detection_retinaface_ensemble", inputs=in_retinaface, outputs=out_retinaface)
# 		croped_image = results.as_numpy("croped_image")
# 		if len(croped_image)==0:
# 			return {"success": False, "error_code": 8001, "error": "Don't find any face"}
# 		#---------------spoofing--------------
# 		result = SPOOFINGDET.inference([img])[0]
# 		print("---------result_spoofing", result)
# 		if result[1] > 0.85:
# 			# img_list = os.listdir("./image_spoofing")
# 			# cv2.imwrite(f"./image_spoofing/{len(img_list)}.jpg", img)
# 			return {"success": False, "error_code": 8002, "error": "Fake face image"}
# 		return {"success": True}
# 		#//////////////////////////////////////
# 	except Exception as e:
# 		return {"success": False, "error": str(e)}

@app.post('/healthcheck')
async def health_check():
	return { 'success': True, 'message': "healthy" }

if __name__=="__main__":
	host = "0.0.0.0"
	port = 8423

	uvicorn.run("controller2:app", host=host, port=port, log_level="info", reload=False)


"""
8000: "Don't have any registered user"
8001: "Don't find any face"
8002: "Fake face image"
8003: "Face photo and identification photo is not similar!"
8004: "Some parts of your face are covered up"
8005: "You are wearing glasses!"
8006: "You are wearing mask!"
8007: "Too many faces in this image"
8008: error system
8009: "Face size is not true"
"""


# docker run -it --shm-size=4g --rm -p8000:8000 -p8001:8001 -p8002:8002 -e PYTHONIOENCODING=UTF-8 -v ${PWD}:/workspace/ -v ${PWD}/my_repository:/models -v ${PWD}/requirements.txt:/opt/tritonserver/requirements.tx tritonserver_mq

# tritonserver --model-repository=/models --model-control-mode=poll --repository-poll-secs=5
