#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/set_env.sh

$(aws ecr get-login --no-include-email --region us-west-2  --registry-ids 763104351884)
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} $DIR/..
