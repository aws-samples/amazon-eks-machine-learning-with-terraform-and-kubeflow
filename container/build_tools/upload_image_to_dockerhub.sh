#!/usr/bin/env bash

source ./set_env.sh

docker login --username=${DH_ACCOUNT}
docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${DH_ACCOUNT}/${IMAGE_NAME}:${IMAGE_TAG}
docker push ${DH_ACCOUNT}/${IMAGE_NAME}