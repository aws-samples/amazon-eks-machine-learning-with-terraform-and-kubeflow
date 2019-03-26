#!/bin/bash

# Customize S3_BUCKET
S3_BUCKET=

# Customize S3_PREFIX
S3_PREFIX=mask-rcnn/eks/input

# Uncomment one of the options below
# For EFS uncomment below
DATA_DIR=/efs
# For FSX uncomment below
#DATA_DIR=/fsx
# For EBS uncomment below
#DATA_DIR=$HOME


if [ -e $DATA_DIR/data ]
then
echo "$DATA_DIR/data already exists"
exit 1
fi

mkdir -p $DATA_DIR/data

aws s3 cp --recursive s3://$S3_BUCKET/$S3_PREFIX/data $DATA_DIR/data

if [ -f ./run.sh ] 
then
	cp run.sh $DATA_DIR/
fi
