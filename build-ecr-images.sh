#!/bin/bash

# set region
region=
if [ "$#" -eq 1 ]; then
    region=$1
else
    echo "usage: $0 <aws-region>"
    exit 1
fi

cd containers
for dir in `ls -d *`
do
$dir/build_tools/build_and_push.sh $region
done
