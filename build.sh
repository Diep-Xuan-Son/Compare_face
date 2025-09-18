#!/bin/bash
IMAGE_NAME=dixuson/compare_face_obfuscated:1.1
docker build -t $IMAGE_NAME -f Dockerfile_controller_obfuscated .
docker image push $IMAGE_NAME