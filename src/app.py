from enum import Enum
from starlette.types import ASGIApp
from starlette.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import FastAPI, Request, Depends, Body, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.exceptions import HTTPException as StarletteHTTPException

from schemes import *
from libs.utils import *
from libs.secret import *
# from triton_services import *

from service_ai.cascade import Cascade
from service_ai.detect_face import DetectFace
from service_ai.check_quality import ImageQuality
from service_ai.detect_mask import MaskClasifyOnnx
from service_ai.spoof_detection_onnx import FakeFace
from service_ai.detect_glasses import GlassesDetection
from service_ai.retinanet_det import RetinanetRunnable
from service_ai.ghostface_onnx import GhostFaceRunnable
from service_ai.arcface_onnx_facereg import FaceRegRunnable
# from service_ai.spoof_detection import SpoofDetectionRunnable

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
PATH_IMG_FAIL = f"{str(ROOT)}/static/failface"
PATH_LOG = f"{str(ROOT)}/logs"
check_folder_exist(path_imgfail=PATH_IMG_FAIL)


LOGGER_APP= logger.bind(name="logger_app")

LOGGER_APP.add(os.path.join(PATH_LOG, f"compareface.{dtdate.today()}.log"), mode='w')

CONFIG_FACEDET = {
	"model_path": f"{str(ROOT)}/weights/detectFace_model_op16.onnx",
	"min_sizes": [[16,32], [64,128], [256,512]],
	"steps": [8, 16, 32],
	"variance": [0.1, 0.2],
	"clip": False,
	"conf_thres": 0.75,
	"iou_thres": 0.25,
	"image_size": [640,640],
	"device": "cpu",
}


CONFIG_FACEREG = {
	"model_path": f"{str(ROOT)}/weights/mxnet_regFace.onnx",
	"imgsz": [112,112],
	"conf_thres": 0.75,
	"device": 'cpu',
}

CONFIG_GHOSTFACE = {
	"model_path": f"{str(ROOT)}/weights/ghostface.onnx",
	"imgsz": [112,112],
	"conf_thres": 0.75,
	"device": 'cpu',
}


FACEDET = RetinanetRunnable(**CONFIG_FACEDET)
FACEREG = FaceRegRunnable(**CONFIG_FACEREG)
CCA = Cascade()
IMGQUALITY = ImageQuality()
SPOOFINGDET = FakeFace(f"{str(ROOT)}/weights/spoofing.onnx")
MASKCLASIFY = MaskClasifyOnnx(f"{str(ROOT)}/weights/mask_detector.onnx")
GLASSESDET = GlassesDetection(f"{str(ROOT)}/weights/glasses_seg.onnx")
DTF = DetectFace(f"{str(ROOT)}/weights/res10_300x300_ssd_iter_140000.caffemodel", f"{str(ROOT)}/weights/deploy.prototxt")
GHOSTFACE = GhostFaceRunnable(**CONFIG_GHOSTFACE)

# TRITONSERVER_IP = os.getenv('TRITONSERVER_IP', "192.168.6.163")
# TRITONSERVER_PORT = os.getenv('TRITONSERVER_PORT', 8001)
# REDISSERVER_IP = os.getenv('REDISSERVER_IP', "192.168.6.142")
# REDISSERVER_PORT = os.getenv('REDISSERVER_PORT', 6400)
# print("----TRITONSERVER_IP: ", TRITONSERVER_IP)
# print("----TRITONSERVER_PORT: ", TRITONSERVER_PORT)
# print("----REDISSERVER_IP: ", REDISSERVER_IP)
# print("----REDISSERVER_PORT: ", REDISSERVER_PORT)
# tritonClient = get_triton_client_aio(ip_address=f"{TRITONSERVER_IP}:{TRITONSERVER_PORT}")
# tritonClient = get_triton_client(ip_address=f"{TRITONSERVER_IP}:{TRITONSERVER_PORT}")
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
	"https://localhost",
	"http://localhost:8422",
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

# @app.exception_handler(StarletteHTTPException)
# async def http_exception_handler(request: Request, exc: StarletteHTTPException):
# 	# Do some logging here
# 	print('aaaaa')
# 	print(exc.detail)
# 	return JSONResponse(content={"detail (specify as desired)": exc.detail}, status_code=exc.status_code)

@app.middleware("http")
async def exception_handling_middleware(request: Request, call_next):
	try:
		return await call_next(request)
	except Exception as e:
		# return  JSONResponse(content=f"Internal Server Error: {str(e)}", status_code=500)
		raise e
		# except AttributeError as a:
	# 	# Do some logging here
	# 	print(str(Exception))
	# 	return JSONResponse(content=f"Internal Server Error", status_code=500)
