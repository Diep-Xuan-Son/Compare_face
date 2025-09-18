import asyncio
import uvicorn
# import redis
import threading
from io import BytesIO
import multiprocessing
import traceback

from app import *

heart_beat_thread = threading.Thread(target=delete_file_cronj, args=(PATH_IMG_FAIL, 25200), daemon=True)
heart_beat_thread.start()

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
	try:
		image_byte = await image_face.read()
		nparr = np.frombuffer(image_byte, np.uint8)
		img_face = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

		image_byte = await image_identification.read()
		nparr = np.frombuffer(image_byte, np.uint8)
		img_id = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
		st_time = time.time()
		#---------------------------face det-------------------------
		t_det = time.time()
		#--------------detect face method 3----------
		results = await asyncio.gather(*[
				asyncio.to_thread(FACEDET.inference, [img_face]),
				asyncio.to_thread(FACEDET.inference, [img_id])
			])

		dets_face, miss_det, croped_image_face = results[0]
		dets_id, miss_det, croped_image_id = results[1]

		if len(dets_face) == 0:
			return {"success": False, "error_code": 8001, "error": "Không tìm thấy khuôn mặt nào trong ảnh chụp", "error_en": "Don't find any face in captured photo"}
		box_face = dets_face[0]["loc"]
		box_face = box_face.astype(int)
		w_crop = (box_face[2]-box_face[0])
		h_crop = (box_face[3]-box_face[1])

		if len(croped_image_id)==0:
			return {"success": False, "error_code": 8001, "error": "Không tìm thấy bất kỳ khuôn mặt nào trong ảnh CCCD", "error_en": "Don't find any face in identification photo"}
		#////////////////////////////////////////////
		LOGGER_APP.info(f"----Duration det: {time.time()-t_det}")

		t_det = time.time()
		box_expand = np.array([max(box_face[0]-w_crop,0), max(box_face[1]-h_crop,0), min(box_face[2]+w_crop, img_face.shape[1]), min(box_face[3]+h_crop, img_face.shape[0])], dtype=int)
		# check_results = await asyncio.gather(*[
		# 		IMGQUALITY.infer(img_face[box_face[1]:box_face[3], box_face[0]:box_face[2]]),
		# 		SPOOFINGDET.inference([img_face[box_expand[1]:box_expand[3], box_expand[0]:box_expand[2]]]),
		# 		GLASSESDET.infer(img_face, box_face.astype(int)),
		# 		MASKCLASIFY(img_face[box_face[1]:box_face[3], box_face[0]:box_face[2]]),
		# 		CCA.infer(croped_image_face[0]),
		# 	])
		check_results = await asyncio.gather(*[
				asyncio.to_thread(SPOOFINGDET.inference, [img_face[box_expand[1]:box_expand[3], box_expand[0]:box_expand[2]]]),
				asyncio.to_thread(MASKCLASIFY, img_face[box_face[1]:box_face[3], box_face[0]:box_face[2]]),
				asyncio.to_thread(IMGQUALITY.infer, img_face[box_face[1]:box_face[3], box_face[0]:box_face[2]]),
				asyncio.to_thread(GLASSESDET.infer, img_face, box_face.astype(int)),
				CCA.infer(croped_image_face[0]),
			])
		LOGGER_APP.info(f"----{check_results}")
		LOGGER_APP.info(f"----Duration face check: {time.time()-t_det}")
		spoofing = check_results[0][0]
		LOGGER_APP.info(f"----result_spoofing: {spoofing}")
		is_mask = check_results[1]
		quality_mes, quality_mes_en = check_results[2]
		LOGGER_APP.info(f"----quality_mes: {quality_mes}")
		is_glasses = check_results[3]
		check_eye_cover, check_nose_cover, check_mouth_cover, part_cover, part_cover_en = check_results[4]
		if spoofing[1] > 0.8499:
			return {"success": False, "error_code": 8002, "error": "Ảnh giả mạo"+quality_mes, "error_en": "Fake face image"+quality_mes_en}
		if is_glasses:
			return {"success": False, "error_code": 8005, "error": f"Khuôn mặt trong ảnh chụp bị che bởi kính"+quality_mes, "error_en": "You are wearing glasses!"+quality_mes_en}
		if is_mask:
			return {"success": False, "error_code": 8006, "error": f"Khuôn mặt trong ảnh chụp bị che bởi khẩu trang"+quality_mes, "error_en": "You are wearing mask!"+quality_mes_en}
		if check_eye_cover or check_nose_cover or check_mouth_cover:
			return {"success": False, "error_code": 8004, "error": f"{part_cover} bị che"+quality_mes, "error_en": f"{part_cover_en} were covered up"+quality_mes_en}
		#////////////////////////////////////////////////////////////
		
		t_reg = time.time()
		# #---------------------------face reg-------------------------
		features = await asyncio.gather(*[
				asyncio.to_thread(GHOSTFACE.get_feature_without_det, croped_image_face),
				asyncio.to_thread(GHOSTFACE.get_feature_without_det, croped_image_id)
			])
		feature_face = features[0]
		feature_face = np.array(feature_face, dtype=np.float16)
		feature_id = features[1]
		feature_id = np.array(feature_id, dtype=np.float16)
		# #////////////////////////////////////////////////////////////
		LOGGER_APP.info(f"----Duration reg: {time.time()-t_reg}")

		t_comp = time.time()
		#---------------------------compare face----------------------
		# print(feature_face.shape)
		# print(feature_id.shape)
		similarity, similarity_sort_idx = GHOSTFACE.compare_face_1_n_1(feature_face, feature_id)
		similarity = similarity[0]
		LOGGER_APP.info(f"----similarity: {similarity}")
		LOGGER_APP.info(f"----Total time duration: {time.time()-st_time}")
		if similarity < 0.8499:
			name_fail_img = datetime.now().strftime('%Y-%m-%d_%H-%M')
			cv2.imwrite(f'{PATH_IMG_FAIL}/{name_fail_img+"_0"}.jpg', img_face)
			cv2.imwrite(f'{PATH_IMG_FAIL}/{name_fail_img+"_1"}.jpg', img_id)
			return {"success": False, "error_code": 8003, "error": "Khuôn mặt người dùng và khuôn mặt trên CCCD không giống nhau"+quality_mes, "error_en": "Face photo and identification photo is not similar!"+quality_mes_en}
		return {"success": True, "similarity": float(similarity), "warning": quality_mes.strip(" ."), "warning_en": quality_mes_en.strip(" .")}
		#/////////////////////////////////////////////////////////////
	except Exception as e:
		tb_str = traceback.format_exc()
		LOGGER_APP.error(f"Traceback: {tb_str}")
		return {"success": False, "error": str(e)}

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
		if result[1] > 0.85:
			# img_list = os.listdir("./image_spoofing")
			# cv2.imwrite(f"./image_spoofing/{len(img_list)}.jpg", img)
			return {"success": False, "error_code": 8002, "error": "Fake face image"}
		return {"success": True}
		# //////////////////////////////////////
	except Exception as e:
		return {"success": False, "error": str(e)}

@app.get('/healthcheck')
async def health_check():
	return { 'success': True, 'message': "healthy" }

if __name__=="__main__":
	host = "0.0.0.0"
	port = 8422
	num_core = multiprocessing.cpu_count()

	uvicorn.run("controller:app", host=host, port=port, log_level="info", reload=False, workers=1, limit_concurrency=num_core, limit_max_requests=num_core*2)


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