#!/bin/bash

# Customize S3_BUCKET
S3_BUCKET=

# Customize S3_PREFIX
S3_PREFIX=mask-rcnn/eks/input

# Customize Stage DIR
# Stage directory must be on EBS volume with 100 GB available space
STAGE_DIR=$HOME/stage

if [ -e $STAGE_DIR ]
then
echo "$STAGE_DIR already exists"
exit 1
fi

mkdir -p $STAGE_DIR/data 

wget -O $STAGE_DIR/data/train2017.zip http://images.cocodataset.org/zips/train2017.zip
unzip $STAGE_DIR/data/train2017.zip  -d $STAGE_DIR/data
rm $STAGE_DIR/data/train2017.zip

wget -O $STAGE_DIR/data/val2017.zip http://images.cocodataset.org/zips/val2017.zip
unzip $STAGE_DIR/data/val2017.zip -d $STAGE_DIR/data
rm $STAGE_DIR/data/val2017.zip

wget -O $STAGE_DIR/data/annotations_trainval2017.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip $STAGE_DIR/data/annotations_trainval2017.zip -d $STAGE_DIR/data
rm $STAGE_DIR/data/annotations_trainval2017.zip

mkdir $STAGE_DIR/data/pretrained-models
wget -O $STAGE_DIR/data/pretrained-models/ImageNet-R50-AlignPadding.npz http://models.tensorpack.com/FasterRCNN/ImageNet-R50-AlignPadding.npz

aws s3 cp --recursive $STAGE_DIR/data s3://$S3_BUCKET/$S3_PREFIX/data
