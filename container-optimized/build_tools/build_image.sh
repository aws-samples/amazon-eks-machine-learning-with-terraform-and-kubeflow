#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/set_env.sh

docker build -t ${IMAGE_NAME}:${IMAGE_TAG} $DIR/..
