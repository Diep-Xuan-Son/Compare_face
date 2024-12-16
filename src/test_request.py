import numpy as np
import cv2
import torch
import tritonclient.grpc as grpcclient
import requests
import json
from locust import HttpUser, task

class NumpyArrayEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.integer):
			return int(obj)
		elif isinstance(obj, np.floating):
			return float(obj)
		elif isinstance(obj, np.ndarray):
			return obj.tolist()
		else:
			return super(NumpyArrayEncoder, self).default(obj)


host = "http://192.168.6.163:8422/api/compareFace"
headers = {'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c3IiOiJjb21wYXJlZmFjZSIsInBhc3MiOiJhaU1RQDIwMTMwNTE2IiwiZXhwIjoxNzMzMzg2NTI5fQ.YjBrBz4EFob63bNXbPuSuDWHoAqETuRpHBemyGGRiCg'
            }
# img = cv2.imread("hieu.jpg")
# files = [("image_face", ("0.jpg", open("image_test/0.jpg", "rb"), 'image/jpeg')), ("image_identification", ("0.jpg", open("image_test/0.jpg", "rb"), 'image/jpeg'))]


# response = requests.post(
#         host,
#         # cookies=cookies,
#         # data=json.dumps(data, cls=NumpyArrayEncoder),
#         files=files,
#         headers=headers
#     )
# print(response)
# print(response.json())
# exit()

class ModelUser(HttpUser):
    @task
    def detect(self):
        files = [("image_face", ("0.jpg", open("image_test/0.jpg", "rb"), 'image/jpeg')), ("image_identification", ("0.jpg", open("image_test/0.jpg", "rb"), 'image/jpeg'))]
        self.client.post(
            "/api/compareFace",
            # cookies=cookies,
            # data=json.dumps(data, cls=NumpyArrayEncoder),
            files=files,
            headers=headers,
        )
