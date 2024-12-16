import os, sys
from pathlib import Path
import json 
import numpy as np
import cv2
from io import BytesIO
import shutil
import threading
import multiprocessing
import asyncio
import uvicorn
# import redis
import time
# from gunicorn.app.base import BaseApplication
# from typing import Any, Callable, Dict

from enum import Enum
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import FastAPI, Request, Depends, Body, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from schemes import *
from triton_services import *
from libs.utils import *
from libs.secret import *

from service_ai.spoof_detection import SpoofDetectionRunnable
from service_ai.spoof_detection_onnx import FakeFace
from service_ai.cascade import Cascade
from service_ai.detect_mask import MaskClasifyOnnx
from service_ai.detect_glasses import GlassesDetection
from service_ai.check_quality import ImageQuality
from service_ai.detect_face import DetectFace

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

# SPOOFINGDET = SpoofDetectionRunnable(**{"model_path": f"{str(ROOT)}/weights/spoofing.pt",
# 									"imgsz": 448,
# 									"device": 'cpu',
# 									"cls_names": ['authentic', 'fake']})
SPOOFINGDET = FakeFace(f"{str(ROOT)}/weights/spoofing.onnx")
CCA = Cascade()
MASKCLASIFY = MaskClasifyOnnx(f"{str(ROOT)}/weights/mask_detector.onnx")
GLASSESDET = GlassesDetection(f"{str(ROOT)}/weights/segmentation_full_lraspp_mobilenet_v3_large.pth")
IMGQUALITY = ImageQuality()
DTF = DetectFace(f"{str(ROOT)}/weights/res10_300x300_ssd_iter_140000.caffemodel", f"{str(ROOT)}/weights/deploy.prototxt")

TRITONSERVER_IP = os.getenv('TRITONSERVER_IP', "192.168.6.163")
TRITONSERVER_PORT = os.getenv('TRITONSERVER_PORT', 8001)
# REDISSERVER_IP = os.getenv('REDISSERVER_IP', "192.168.6.142")
# REDISSERVER_PORT = os.getenv('REDISSERVER_PORT', 6400)
print("----TRITONSERVER_IP: ", TRITONSERVER_IP)
print("----TRITONSERVER_PORT: ", TRITONSERVER_PORT)
# print("----REDISSERVER_IP: ", REDISSERVER_IP)
# print("----REDISSERVER_PORT: ", REDISSERVER_PORT)
tritonClient = get_triton_client(ip_address=f"{TRITONSERVER_IP}:{TRITONSERVER_PORT}")
# redisClient = redis.StrictRedis(host=REDISSERVER_IP,
# 								port=int(REDISSERVER_PORT),
# 								password="RedisAuth",
# 								db=0)

class FileTypeName(str, Enum):
	jpeg = "image/jpeg"
	jpg = "image/jpg"
	png = "image/png"
	# bmp = "image/bmp"
	# webp = "image/webp"

	@classmethod
	def list(cls):
		return list(map(lambda c: c.value, cls))

class ValidateUploadFileMiddleware(BaseHTTPMiddleware):
	def __init__(
		self,
		app: ASGIApp, max_size: int = 10048576, # 10MB
		file_types: List[str] = FileTypeName.list()
	) -> None:
		super().__init__(app)
		self.max_size = max_size
		self.file_types = file_types

	async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
		if request.method == 'POST':
			if request.headers['content-type'] in ["multipart/form-data"]:
				form = await request.form()
				try:
					content_type = form[next(iter(form))].content_type
				except:
					content_type = self.file_types[0]
				if content_type not in self.file_types:
					return Response(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)
			if 'content-length' not in request.headers:
				return Response(status_code=status.HTTP_411_LENGTH_REQUIRED)
			content_length = int(request.headers['content-length'])
			if content_length > self.max_size:
				return Response(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)
		return await call_next(request)

app = FastAPI()
origins = [
	"http://localhost",
	"http://localhost:8080",
]
app.add_middleware(
	CORSMiddleware,
	allow_origins=origins,
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)
app.add_middleware(
	ValidateUploadFileMiddleware,
)
# app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/hash_password")
async def get_passwd_hash(passwd: str = Body()):
	return get_password_hash(passwd)

@app.post(TOKEN_URL)
async def login_for_access_token(
	form_data: OAuth2PasswordRequestForm = Depends(),
) -> Token:
	user = authenticate_user(fake_users_db, form_data.username, form_data.password)
	if not user:
		raise HTTPException(
			status_code=status.HTTP_401_UNAUTHORIZED,
			detail="Incorrect username or password",
			headers={"WWW-Authenticate": "Bearer"},
		)
	access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
	access_token = create_access_token(
		data={"usr": user.username, "pass": form_data.password}, expires_delta=access_token_expires
	)
	return Token(access_token=access_token, token_type="bearer")

@app.get("/users/me/", response_model=User)
async def read_users_me(token: str,):
	current_user: User = await get_current_active_user(token=token)
	return current_user

@app.get("/get_token")
async def read_items(token: str = Depends(oauth2_scheme)):
	return {"token": token}

@app.post("/api/compareFace")
async def compareFace(image_face: UploadFile = File(...), image_identification: UploadFile = File(...), token: str = Depends(oauth2_scheme)):
	# try:
	image_byte = await image_face.read()
	nparr = np.fromstring(image_byte, np.uint8)
	img_face = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

	image_byte = await image_identification.read()
	nparr = np.fromstring(image_byte, np.uint8)
	img_id = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
	st_time = time.time()
	#---------------------------face det-------------------------
	t_det = time.time()
	in_retinaface_face, out_retinaface_face = get_io_retinaface(img_face)
	# results = await tritonClient.infer(model_name="detection_retinaface_ensemble", inputs=in_retinaface_face)
	# croped_image_face = results.as_numpy("croped_image")
	# if len(croped_image_face)==0:
	# 	return {"success": False, "error_code": 8001, "error": "Don't find any face in face photo"}
	# box_face = results.as_numpy("box")[0]
	# box_face = box_face.astype(int)
	# w_crop = (box_face[2]-box_face[0])
	# h_crop = (box_face[3]-box_face[1])

	in_retinaface_id, out_retinaface_id = get_io_retinaface(img_id)
	# results = await tritonClient.infer(model_name="detection_retinaface_ensemble", inputs=in_retinaface)
	# features = await asyncio.gather(*[
	# 		tritonClient.infer(model_name="detection_retinaface_ensemble", inputs=in_retinaface_face),
	# 		tritonClient.infer(model_name="detection_retinaface_ensemble", inputs=in_retinaface_id)
	# 	])
	results = await asyncio.gather(*[
			asyncio.to_thread(tritonClient.infer, **{"model_name":"detection_retinaface_ensemble", "inputs":in_retinaface_face}),
			asyncio.to_thread(tritonClient.infer, **{"model_name":"detection_retinaface_ensemble", "inputs":in_retinaface_id})
		])
	result_face = results[0]
	result_id = results[1]

	croped_image_face = result_face.as_numpy("croped_image")
	if len(croped_image_face)==0:
		return {"success": False, "error_code": 8001, "error": "Don't find any face in face photo"}
	box_face = result_face.as_numpy("box")[0]
	box_face = box_face.astype(int)
	w_crop = (box_face[2]-box_face[0])
	h_crop = (box_face[3]-box_face[1])

	croped_image_id = result_id.as_numpy("croped_image")
	if len(croped_image_id)==0:
		return {"success": False, "error_code": 8001, "error": "Don't find any face in identification photo"}
	
	# #---------detect face method 2---------
	# results = await asyncio.gather(*[
	# 		DTF.infer(img_face),
	# 		DTF.infer(img_id)
	# 	])
	# croped_image_face, box_face = results[0]
	# croped_image_id, box_id = results[1]

	# if len(croped_image_face)==0:
	# 	return {"success": False, "error_code": 8001, "error": "Don't find any face in face photo"}
	# box_face = box_face.astype(int)
	# w_crop = (box_face[2]-box_face[0])
	# h_crop = (box_face[3]-box_face[1])

	# if len(croped_image_id)==0:
	# 	return {"success": False, "error_code": 8001, "error": "Don't find any face in identification photo"}
	# #////////////////////////////////////////
	print("------Duration det: ", time.time()-t_det)
	# #-----------image quality------------
	# t_det = time.time()
	# quality_mes = IMGQUALITY.infer(img_face[box_face[1]:box_face[3], box_face[0]:box_face[2]])
	# print("----quality_mes: ", quality_mes)
	# print("------Duration quality: ", time.time()-t_det)
	# #/////////////////////////////////////
	# #---------------spoofing--------------
	# t_det = time.time()
	# box_expand = np.array([max(box_face[0]-w_crop,0), max(box_face[1]-h_crop,0), min(box_face[2]+w_crop, img_face.shape[1]), min(box_face[3]+h_crop, img_face.shape[0])], dtype=int)
	# result = SPOOFINGDET.inference([img_face[box_expand[1]:box_expand[3], box_expand[0]:box_expand[2]]])
	# result = result[0]
	# print("---------result_spoofing", result)
	# if result[1] > 0.85:
	# 	# img_list = os.listdir("./image_test")
	# 	# cv2.imwrite(f"./image_test/{len(img_list)}.jpg", img_spoofing)
	# 	return {"success": False, "error_code": 8002, "error": "Fake face image"+quality_mes}
	# print("------Duration spoofing: ", time.time()-t_det)
	# #//////////////////////////////////////

	# #---------------glasses----------------
	# t_det = time.time()
	# is_glasses = GLASSESDET.infer(img_face, box_face.astype(int))
	# if is_glasses:
	# 	return {"success": False, "error_code": 8005, "error": f"Face is convered up by glasses!"+quality_mes}
	# print("------Duration glasses: ", time.time()-t_det)
	# #//////////////////////////////////////

	# #------------------mask----------------
	# t_det = time.time()
	# is_mask = MASKCLASIFY(img_face[box_face[1]:box_face[3], box_face[0]:box_face[2]])
	# if is_mask:
	# 	return {"success": False, "error_code": 8006, "error": f"Face is convered up by mask!"+quality_mes}
	# print("------Duration mask: ", time.time()-t_det)
	# #//////////////////////////////////////

	# #---------------cascade----------------
	# t_det = time.time()
	# # print(croped_image_face.shape)
	# # box_expand = np.array([max(box_face[0]-w_crop/4,0), max(box_face[1]-h_crop/4,0), min(box_face[2]+w_crop/4, img_face.shape[1]), min(box_face[3]+h_crop/4, img_face.shape[0])], dtype=int)
	# check_eye_cover, check_nose_cover, check_mouth_cover, part_cover = CCA.infer(croped_image_face[0])
	# if check_eye_cover or check_nose_cover or check_mouth_cover:
	# 	return {"success": False, "error_code": 8004, "error": f"{part_cover} was covered up"+quality_mes}
	# print("------Duration cascade: ", time.time()-t_det)
	# #//////////////////////////////////////
	t_det = time.time()
	box_expand = np.array([max(box_face[0]-w_crop,0), max(box_face[1]-h_crop,0), min(box_face[2]+w_crop, img_face.shape[1]), min(box_face[3]+h_crop, img_face.shape[0])], dtype=int)
	check_results = await asyncio.gather(*[
			asyncio.to_thread(IMGQUALITY.infer, img_face[box_face[1]:box_face[3], box_face[0]:box_face[2]]),
			asyncio.to_thread(SPOOFINGDET.inference, [img_face[box_expand[1]:box_expand[3], box_expand[0]:box_expand[2]]]),
			asyncio.to_thread(GLASSESDET.infer, img_face, box_face.astype(int)),
			asyncio.to_thread(MASKCLASIFY, img_face[box_face[1]:box_face[3], box_face[0]:box_face[2]]),
			asyncio.to_thread(CCA.infer, croped_image_face[0]),
		])
	print(check_results)
	print("------Duration face check: ", time.time()-t_det)
	quality_mes = check_results[0]
	print("----quality_mes: ", quality_mes)
	spoofing = check_results[1][0]
	print("---------result_spoofing", spoofing)
	is_glasses = check_results[2]
	is_mask = check_results[3]
	check_eye_cover, check_nose_cover, check_mouth_cover, part_cover = check_results[4]
	if spoofing[1] > 0.85:
		return {"success": False, "error_code": 8002, "error": "Fake face image"+quality_mes}
	if is_glasses:
		return {"success": False, "error_code": 8005, "error": f"Face is convered up by glasses!"+quality_mes}
	if is_mask:
		return {"success": False, "error_code": 8006, "error": f"Face is convered up by mask!"+quality_mes}
	if check_eye_cover or check_nose_cover or check_mouth_cover:
		return {"success": False, "error_code": 8004, "error": f"{part_cover} was covered up"+quality_mes}
	#////////////////////////////////////////////////////////////
	
	t_reg = time.time()
	#---------------------------face reg-------------------------
	in_ghostface_face, out_ghostface_face = get_io_ghostface(croped_image_face)
	# results = await tritonClient.infer(model_name="ghost_face_nodet_ensemble", inputs=in_ghostface, outputs=out_ghostface)
	# feature_face = results.as_numpy("feature_norm")
	# feature_face = feature_face.astype(np.float16)

	in_ghostface_id, out_ghostface_id = get_io_ghostface(croped_image_id)
	# results = await tritonClient.infer(model_name="ghost_face_nodet_ensemble", inputs=in_ghostface, outputs=out_ghostface)
	# features = await asyncio.gather(*[
	# 		tritonClient.infer(model_name="ghost_face_nodet_ensemble", inputs=in_ghostface_face, outputs=out_ghostface_face),
	# 		tritonClient.infer(model_name="ghost_face_nodet_ensemble", inputs=in_ghostface_id, outputs=out_ghostface_id)
	# 	])
	features = await asyncio.gather(*[
			asyncio.to_thread(tritonClient.infer, **{"model_name":"ghost_face_nodet_ensemble", "inputs":in_ghostface_face, "outputs":out_ghostface_face}),
			asyncio.to_thread(tritonClient.infer, **{"model_name":"ghost_face_nodet_ensemble", "inputs":in_ghostface_id, "outputs":out_ghostface_id})
		])
	feature_face = features[0]
	feature_id = features[1]
	feature_face = feature_face.as_numpy("feature_norm")
	feature_face = feature_face.astype(np.float16)
	feature_id = feature_id.as_numpy("feature_norm")
	feature_id = feature_id.astype(np.float16)
	#////////////////////////////////////////////////////////////
	print("------Duration reg: ", time.time()-t_reg)

	t_comp = time.time()
	#---------------------------compare face----------------------
	print(feature_face.shape)
	print(feature_id.shape)
	dist = np.linalg.norm(feature_face - feature_id, axis=1)
	similarity = (np.tanh((1.23132175 - dist) * 6.602259425) + 1) / 2

	print("---------similarity: ", similarity)
	print("----Total time duration: ", time.time()-st_time)
	if similarity < 0.75:
		return {"success": False, "error_code": 8003, "error": "Face photo and identification photo is not similar"+quality_mes}
	return {"success": True, "similarity": float(similarity), "warning": quality_mes}
	#/////////////////////////////////////////////////////////////
	# except Exception as e:
	# 	return {"success": False, "error": str(e)}
	# except AttributeError as a:
	# 	pass

@app.post("/api/spoofingCheck")
async def spoofingCheck(image: UploadFile = File(...)):
	try:
		image_byte = await image.read()
		# print("ddddd: ", threading.get_ident())
		nparr = np.fromstring(image_byte, np.uint8)
		img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
		#---------------------------face det-------------------------
		in_retinaface, out_retinaface = get_io_retinaface(img)
		results = await tritonClient.infer(model_name="detection_retinaface_ensemble", inputs=in_retinaface, outputs=out_retinaface)
		croped_image = results.as_numpy("croped_image")
		if len(croped_image)==0:
			return {"success": False, "error_code": 8001, "error": "Don't find any face"}
		#---------------spoofing--------------
		result = SPOOFINGDET.inference([img])[0]
		# print("---------result_spoofing", result)
		if result[1] > 0.85:
			# img_list = os.listdir("./image_spoofing")
			# cv2.imwrite(f"./image_spoofing/{len(img_list)}.jpg", img)
			return {"success": False, "error_code": 8002, "error": "Fake face image"}
		return {"success": True}
		# //////////////////////////////////////
		print("bbbbbbbbb")
		return {"success": True}
	except Exception as e:
		return {"success": False, "error": str(e)}

@app.get('/healthcheck')
async def health_check():
	return { 'success': True, 'message': "healthy" }

if __name__=="__main__":
	host = "0.0.0.0"
	port = 8422
	num_core = multiprocessing.cpu_count()

	uvicorn.run("controller:app", host=host, port=port, log_level="info", reload=False, workers=1, limit_concurrency=num_core*2, limit_max_requests=num_core*2)


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

# uvicorn --workers 1 --host 0.0.0.0 --port 8422 controller:app