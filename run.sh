#!/bin/bash

echo "Loading image ..."
docker load -i compare_face_ob.tar

echo "Running service ..."
docker compose -f docker-compose.yml --profile controller_ai_obfuscated up -d