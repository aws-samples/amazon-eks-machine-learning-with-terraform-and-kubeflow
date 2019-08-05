#!/usr/bin/env bash

source ./set_env.sh

docker build -t ${IMAGE_NAME}:${IMAGE_TAG} ..
