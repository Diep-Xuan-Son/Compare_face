from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

import multiprocessing as mp
import time
import requests

class MyClass():
	def __init__(self):
		self.text = "hello s"

	def execute(self, proid=0):
		time.sleep(3)
		print(f"{proid}: {self.text}")
		return f"result {proid}: {self.text}"

class ProcessWorker(mp.Process):
	"""
	This class runs as a separate process to execute worker's commands in parallel
	Once launched, it remains running, monitoring the task queue, until "None" is sent
	"""

	def __init__(self, task_q, result_q):
		mp.Process.__init__(self)
		self.task_q = task_q
		self.result_q = result_q
		return

	def run(self):
		"""
		Overloaded function provided by multiprocessing.Process.  Called upon start() signal
		"""
		proc_name = self.name
		print(f'{proc_name}: Launched')
		while True:
			print("---self.task_q: ", self.task_q)
			next_task_list = self.task_q.get()
			print("---next_task_list: ", next_task_list)
			if next_task_list is None:
				# Poison pill means shutdown
				print(f'{proc_name}: Exiting')
				self.task_q.task_done()
				break
			next_task = next_task_list[0]
			print(f'{proc_name}: {next_task}')
			args = next_task_list[1]
			kwargs = next_task_list[2]
			answer = next_task(*args, **kwargs)
			self.task_q.task_done()
			self.result_q.put({proc_name: answer})
			time.sleep(1e-6)
		return
# End of ProcessWorker class

class BaseWorker(mp.Process):
	"""
	Launches a child process to run commands from derived classes in separate processes,
	which sit and listen for something to do
	This base class is called by each derived worker
	"""
	def __init__(self, conn, ready_event, shutdown_event, interrupt_stop_event):
		mp.Process.__init__(self)
		self.conn = conn
		self.ready_event = ready_event
		self.shutdown_event = shutdown_event
		self.interrupt_stop_event = interrupt_stop_event
		self.inp_queue = mp.Queue()
		self.out_queue = mp.Queue()

		self.time_sleep = 1e-4
		# self.process_worker = ProcessWorker(self.task_q, self.result_q)
		# self.proc_name = self.process_worker.name
		# self.funtion_kwarg["proid"] = self.proc_name
		# print(self.process_worker)
		# exit()
		# self.process_worker.start()
		# print("Got here")
		# Process should be running and listening for functions to execute
		return

	def poll_connection(self):
		while not self.shutdown_event.is_set():
			if self.conn.poll(timeout=0.1):    #This will return a boolean as to whether there is data to be received and read from the pipe
				try:
					data = self.conn.recv()
					self.inp_queue.put(data)
				except Exception as e:
					logging.error(f"Error receiving data from worker connection: {e}")
			else:
				time.sleep(self.time_sleep)

	def close(self):
		self.conn.close()
		self.shutdown_event.set() # Ensure the polling thread will stop
		self.join()

from service_ai.spoof_detection import SpoofDetectionRunnable
from service_ai.spoof_detection_onnx import FakeFace
from service_ai.cascade import Cascade
from service_ai.detect_mask import MaskClasifyOnnx
from service_ai.detect_glasses import GlassesDetection
from service_ai.check_quality import ImageQuality
from triton_services import *
import numpy as np
import threading
import logging
import queue
import cv2
class CompareFaceWorker(BaseWorker):
	def __init__(self, conn, ready_event, shutdown_event, interrupt_stop_event):
		super().__init__(conn, ready_event, shutdown_event, interrupt_stop_event)
		logging.info(f"Initializing models")
		try:
			self.spoofingdet = FakeFace(f"{str(ROOT)}/weights/spoofing.onnx")
			self.cca = Cascade()
			self.maskclasify = MaskClasifyOnnx(f"{str(ROOT)}/weights/mask_detector.onnx")
			self.glassesdet = GlassesDetection(f"{str(ROOT)}/weights/segmentation_full_lraspp_mobilenet_v3_large.pth")
			self.imgquality = ImageQuality()
		except Exception as e:
			logging.exception(f"Error initializing models: {e}")

		TRITONSERVER_IP = os.getenv('TRITONSERVER_IP', "192.168.6.163")
		TRITONSERVER_PORT = os.getenv('TRITONSERVER_PORT', 8001)
		self.tritonClient = get_triton_client(ip_address=f"{TRITONSERVER_IP}:{TRITONSERVER_PORT}")
		self.ready_event.set()
		logging.info("All models for compare face initialized successfully")

	def run(self):
		print("---process id: ", self.name)
		# Start the polling thread
		polling_thread = threading.Thread(target=self.poll_connection, daemon=True)
		polling_thread.start()
		try:
			while not self.shutdown_event.is_set():
				try:
					time.sleep(5)
					input = self.inp_queue.get(timeout=0.1)
					print("---running process: ", self.name)
					img_face = input["img_face"]
					img_id = input["img_id"]
					try:
						t_det = time.time()
						#---------------------------face det-------------------------
						in_retinaface, out_retinaface = get_io_retinaface(img_face)
						results = self.tritonClient.infer(model_name="detection_retinaface_ensemble", inputs=in_retinaface)
						croped_image_face = results.as_numpy("croped_image")
						if len(croped_image_face)==0:
							result = {"success": False, "error_code": 8001, "error": "Don't find any face in face photo"}
							self.out_queue.put(result)
							continue
						box_face = results.as_numpy("box")[0]
						box_face = box_face.astype(int)
						w_crop = (box_face[2]-box_face[0])
						h_crop = (box_face[3]-box_face[1])

						in_retinaface, out_retinaface = get_io_retinaface(img_id)
						results = self.tritonClient.infer(model_name="detection_retinaface_ensemble", inputs=in_retinaface)
						croped_image_id = results.as_numpy("croped_image")
						if len(croped_image_id)==0:
							result = {"success": False, "error_code": 8001, "error": "Don't find any face in identification photo"}
							self.out_queue.put(result)
							continue
						#-----------image quality------------
						quality_mes = self.imgquality.infer(img_face[box_face[1]:box_face[3], box_face[0]:box_face[2]])
						print("----quality_mes: ", quality_mes)
						#/////////////////////////////////////
						#---------------spoofing--------------
						box_expand = np.array([max(box_face[0]-w_crop,0), max(box_face[1]-h_crop,0), min(box_face[2]+w_crop, img_face.shape[1]), min(box_face[3]+h_crop, img_face.shape[0])], dtype=int)
						result = self.spoofingdet.inference([img_face[box_expand[1]:box_expand[3], box_expand[0]:box_expand[2]]])[0]
						print("---------result_spoofing", result)
						if result[1] > 0.85:
							# img_list = os.listdir("./image_test")
							# cv2.imwrite(f"./image_test/{len(img_list)}.jpg", img_spoofing)
							result = {"success": False, "error_code": 8002, "error": "Fake face image"+quality_mes}
							self.out_queue.put(result)
							continue
						#//////////////////////////////////////

						#---------------glasses----------------
						is_glasses = self.glassesdet.infer(img_face, box_face.astype(int))
						print("----is_glasses: ", is_glasses)
						if is_glasses:
							result = {"success": False, "error_code": 8005, "error": f"Face is convered up by glasses!"+quality_mes}
							self.out_queue.put(result)
							continue
						#//////////////////////////////////////

						#------------------mask----------------
						is_mask = self.maskclasify(img_face[box_face[1]:box_face[3], box_face[0]:box_face[2]])
						if is_mask:
							result = {"success": False, "error_code": 8006, "error": f"Face is convered up by mask!"+quality_mes}
							self.out_queue.put(result)
							continue
						#//////////////////////////////////////

						#---------------cascade----------------
						# print(croped_image_face.shape)
						# box_expand = np.array([max(box_face[0]-w_crop/4,0), max(box_face[1]-h_crop/4,0), min(box_face[2]+w_crop/4, img_face.shape[1]), min(box_face[3]+h_crop/4, img_face.shape[0])], dtype=int)
						check_eye_cover, check_nose_cover, check_mouth_cover, part_cover = self.cca.infer(croped_image_face[0])
						if check_eye_cover or check_nose_cover or check_mouth_cover:
							result = {"success": False, "error_code": 8004, "error": f"{part_cover} was covered up"+quality_mes}
							self.out_queue.put(result)
							continue
						#//////////////////////////////////////

						#////////////////////////////////////////////////////////////
						print("------Duration det: ", time.time()-t_det)
						
						t_reg = time.time()
						#---------------------------face reg-------------------------
						in_ghostface, out_ghostface = get_io_ghostface(croped_image_face)
						results = self.tritonClient.infer(model_name="ghost_face_nodet_ensemble", inputs=in_ghostface, outputs=out_ghostface)
						feature_face = results.as_numpy("feature_norm")
						feature_face = feature_face.astype(np.float16)

						in_ghostface, out_ghostface = get_io_ghostface(croped_image_id)
						results = self.tritonClient.infer(model_name="ghost_face_nodet_ensemble", inputs=in_ghostface, outputs=out_ghostface)
						feature_id = results.as_numpy("feature_norm")
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

						if similarity < 0.75:
							result = {"success": False, "error_code": 8003, "error": "Face photo and identification photo is not similar"+quality_mes}
							self.out_queue.put(result)
							continue
						self.out_queue.put({"success": True, "similarity": float(similarity), "warning": quality_mes})
						#/////////////////////////////////////////////////////////////
					except Exception as e:
						logging.error(f"General error in transcription: {e}")
						self.out_queue.put(('error', str(e)))

				except queue.Empty:
					time.sleep(self.time_sleep)
					continue
				except KeyboardInterrupt:
					self.interrupt_stop_event.set()
					logging.debug("Transcription worker process finished due to KeyboardInterrupt")
					break
				except Exception as e:
					logging.error(f"General error in processing queue item: {e}")
		finally:
			self.close()
			polling_thread.join()  # Wait for the polling thread to finish

	# def run(self):
	# 	asyncio.run(self.process())

class MultiWorker(object):
	def __init__(self, 
				 myclass=getattr(__import__(__name__), "MyClass"), 
				 log_level=logging.INFO, 
				 no_log_file: bool = False,
				 ):
		self.num_worker = 0
		self.workers = {}
		self.workers_temp = {}
		self.myclass = myclass
		self.count = 1

		#--------log-------
		# Initialize the logging configuration with the specified level
		log_format = 'CompareFace: %(name)s - %(levelname)s - %(message)s'

		# Adjust file_log_format to include milliseconds
		file_log_format = '%(asctime)s.%(msecs)03d - ' + log_format

		# Get the root logger
		logger = logging.getLogger()
		logger.setLevel(log_level)  # Set the root logger's level to DEBUG

		# Remove any existing handlers
		logger.handlers = []

		# Create a console handler and set its level
		console_handler = logging.StreamHandler()
		console_handler.setLevel(log_level) 
		console_handler.setFormatter(logging.Formatter(log_format))

		# Add the handlers to the logger
		if not no_log_file:
			# Create a file handler and set its level
			file_handler = logging.FileHandler('compareface.log')
			file_handler.setLevel(log_level)
			file_handler.setFormatter(logging.Formatter(
				file_log_format,
				datefmt='%Y-%m-%d %H:%M:%S'
			))

			logger.addHandler(file_handler)
		logger.addHandler(console_handler)
		#//////////////////

	def init_worker(self, num_worker=2):
		for i in range(num_worker):
			self.add_worker()

	def run(self, *args, **kwargs):
		nworker, worker_comp = list(self.workers.items())[0]
		parent_pipe = worker_comp[1]
		parent_pipe.send(kwargs)
		# worker.run(**kwargs)
		self.workers.pop(nworker)
		self.workers[nworker] = worker_comp
		# result = worker.fetch_results()
		# self.workers.append(worker)
		# self.workers.pop(0)
		# return result
		#-----test add worker---------
		self.count += 1
		if self.count%10 == 0:
			self.add_worker()
			res = requests.post("http://localhost:8423/set_available_threads", params={"num_thread": self.num_worker*6})
		#////////////////////////////
		return {"nworker": nworker}

	def get_result(self, proc_name):
		workers_running = {**self.workers, **self.workers_temp}
		worker = workers_running[proc_name][0]
		parent_pipe = workers_running[proc_name][1]
		shutdown_event = workers_running[proc_name][4]
		result = worker.out_queue.get()
		# while not shutdown_event.is_set():
		# 	if parent_pipe.poll(0.1):
		# 		try:
		# 			result = parent_pipe.recv()
		# 		except Exception as e:
		# 			logging.error(f"Error receiving data from parent connection: {e}")
		# 			result = f"Error receiving data from parent connection: {e}"
		# 		break
		# 	else:
		# 		time.sleep(worker.time_sleep)
		return result

	def delete_worker(self, num_worker_dl=1, time_check=1):
		if num_worker_dl>=self.num_worker:
			return f"Too much deleted worker, max number of worker is {self.num_worker}"

		self.num_worker = self.num_worker - num_worker_dl
		for i in range(num_worker_dl):
			k, v = self.workers.popitem()
			self.workers_temp[k] = v

		while True:
			if not num_worker_dl:
				print("----Delete worker done!")
				break
			time.sleep(time_check)
			nworker, worker_comp = list(self.workers_temp.items())[-1]
			worker = worker_comp[0]
			is_queue_empty = worker.out_queue.empty() and worker.inp_queue.empty()
			# print("----is_queue_empty: ", is_queue_empty)
			if is_queue_empty:
				worker.close() 
				self.workers_temp.popitem()
				num_worker_dl -= 1

	def add_worker(self, num_worker_add=1):
		shutdown_event = mp.Event()
		interrupt_stop_event = mp.Event()
		main_ready_event = mp.Event()
		parent_pipe, child_pipe = mp.Pipe()
		args = (child_pipe, main_ready_event, shutdown_event, interrupt_stop_event)

		# worker = self._start_thread(
		# 	target=MultiWorker._run_worker,
		#     args=(args)
		# )
		worker = self.myclass(*args)
		# Wait for models to start
		logging.debug('Waiting for models to start')
		main_ready_event.wait()

		self.workers[worker.name] = (worker, parent_pipe,) + args
		self.num_worker += num_worker_add
		worker.start()

	# def _start_thread(self, target=None, args=()):
	#     """
	#     Implement a consistent threading model across the library.

	#     This method is used to start any thread in this library. It uses the
	#     standard threading. Thread for Linux and for all others uses the pytorch
	#     MultiProcessing library 'Process'.
	#     Args:
	#         target (callable object): is the callable object to be invoked by
	#           the run() method. Defaults to None, meaning nothing is called.
	#         args (tuple): is a list or tuple of arguments for the target
	#           invocation. Defaults to ().
	#     """
	#     # if (platform.system() == 'Linux'):
	#     #     thread = threading.Thread(target=target, args=args)
	#     #     thread.deamon = True
	#     #     thread.start()
	#     #     return thread
	#     # else:
	#     thread = mp.Process(target=target, args=args)
	#     thread.start()
	#     return thread

	# def _run_worker(*args, **kwargs):
	# 	worker = self.myclass(*args, **kwargs)
	#     worker.run()

NUM_WORKER = 2
myapp = MultiWorker(myclass=getattr(__import__(__name__), "CompareFaceWorker"))
myapp.init_worker(num_worker=NUM_WORKER)

if __name__=="__main__":
	NUM_WORKER = 2
	myapp = MultiWorker(myclass=getattr(__import__(__name__), "CompareFaceWorker"))
	myapp.init_worker(num_worker=NUM_WORKER)

	img_face = cv2.imread("./image_test/0.jpg")
	img_id = cv2.imread("./image_test/0.jpg")
	st_time = time.time()
	for i in range(10):
		proc_name = myapp.run(img_face=img_face, img_id=img_id)
		result = myapp.get_result(proc_name=proc_name["nworker"])
		print("----result: ", result)
		myapp.delete_worker()
		exit()
	print("------Duration total: ", time.time()-st_time)