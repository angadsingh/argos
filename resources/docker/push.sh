#!/bin/bash

docker buildx build --platform linux/arm/v7 -t angadsingh/argos:armv7 -f resources/docker/Dockerfile_armv7 . --push
docker build -t angadsingh/argos:x86_64 -f resources/docker/Dockerfile_x86 . && docker push angadsingh/argos:x86_64
docker build -t angadsingh/argos:x86_64_gpu -f resources/docker/Dockerfile_x86_gpu . && docker push angadsingh/argos:x86_64_gpu