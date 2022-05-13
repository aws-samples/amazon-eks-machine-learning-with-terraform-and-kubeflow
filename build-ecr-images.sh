#!/bin/bash

# set region
region=
if [ "$#" -eq 1 ]; then
    region=$1
else
    echo "usage: $0 <aws-region>"
    exit 1
fi

./container/build_tools/build_and_push.sh $region
./container-optimized/build_tools/build_and_push.sh $region
./container-optimized-viz/build_tools/build_and_push.sh $region
